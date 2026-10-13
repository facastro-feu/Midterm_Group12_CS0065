import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_FILE = "lost_found_dataset.csv"

# How much each field counts toward similarity (bigger = more important).
FIELD_WEIGHTS = {
    "category": 3.0,  # a phone should never match a wallet
    "description": 3.0,  # the distinguishing details
    "color": 2.0,
    "brand": 1.5,
    "location": 1.5,
}
DATE_WEIGHT = 0.15  # share of the final score that comes from the date gap
MAX_DAYS_GAP = 30  # found more than 30 days after it was lost -> invalid
MIN_SCORE = 0.40  # below this, report "no strong match"
TOP_K = 5

CATEGORICAL_FIELDS = ["category", "color", "brand", "location"]

# Synonyms that mean the same category.
CATEGORY_MAP = {
    "headphones": "earphones",
    "earbuds": "earphones",
    "bag": "backpack",
    "cellphone": "phone",
    "mobile phone": "phone",
    "hoodie": "jacket",
    "glasses": "eyeglasses",
    "tumbler": "water bottle",
}

# Values that mean "we don't know" - these should not count as a match.
UNKNOWN_VALUES = {"", "unknown", "n/a", "na", "none", "nan"}


# ---------------------------------------------------------------------------
# 1-2. Load and clean
# ---------------------------------------------------------------------------
def clean_text(value):
    """Lowercase, trim, and collapse extra spaces."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def load_and_clean(path=DATA_FILE):
    df = pd.read_csv(path, dtype=str)
    report = {"raw_rows": len(df)}

    # Clean every text field (keep item_id exactly as it is).
    for col in ["record_type"] + CATEGORICAL_FIELDS + ["description"]:
        df[col] = df[col].apply(clean_text)

    # Standardize categories and unknown brands.
    df["category"] = df["category"].replace(CATEGORY_MAP)
    df["brand"] = df["brand"].apply(lambda b: "unknown" if b in UNKNOWN_VALUES else b)

    # Parse dates; invalid dates become NaT and are flagged.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    report["invalid_dates"] = df.loc[df["date"].isna(), "item_id"].tolist()

    # Flag blank descriptions (kept, but they rely on the other fields).
    report["blank_descriptions"] = df.loc[df["description"] == "", "item_id"].tolist()

    # Remove duplicates: same content, different ID.
    content_cols = ["record_type"] + CATEGORICAL_FIELDS + ["date", "description"]
    dup_mask = df.duplicated(subset=content_cols, keep="first")
    report["duplicates_removed"] = df.loc[dup_mask, "item_id"].tolist()
    df = df[~dup_mask].reset_index(drop=True)

    report["clean_rows"] = len(df)
    return df, report


# ---------------------------------------------------------------------------
# 3. Prepare comparable features
# ---------------------------------------------------------------------------
class LostFoundMatcher:
    """Fits encoders on the dataset and ranks recovered items for a query."""

    def __init__(self, df):
        self.df = df
        self.reported = df[df["record_type"] == "reported"].reset_index(drop=True)
        self.recovered = df[df["record_type"] == "recovered"].reset_index(drop=True)

        # One encoder per categorical field. "unknown" is left out of the
        # fitted categories, so it encodes as all zeros (handle_unknown="ignore")
        # and two unknown brands do NOT look similar.
        self.encoders = {}
        for field in CATEGORICAL_FIELDS:
            known = df.loc[~df[field].isin(UNKNOWN_VALUES), [field]]
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit(known)
            self.encoders[field] = enc

        # TF-IDF for the free-text description.
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf.fit(df["description"].replace("", "empty"))

        self.recovered_X = self.vectorize(self.recovered)

    def vectorize(self, rows):
        """Turn rows into one weighted feature matrix.

        Each block (one field) has length 1 before weighting, so multiplying
        by sqrt(weight) makes the field's share of the dot product equal
        to its weight.
        """
        blocks = []
        for field in CATEGORICAL_FIELDS:
            X = self.encoders[field].transform(rows[[field]])
            blocks.append(X * np.sqrt(FIELD_WEIGHTS[field]))
        X_desc = self.tfidf.transform(rows["description"])
        blocks.append(X_desc * np.sqrt(FIELD_WEIGHTS["description"]))
        return csr_matrix(hstack(blocks))

    # -----------------------------------------------------------------------
    # 5. Filter invalid items
    # -----------------------------------------------------------------------
    def valid_candidates(self, query):
        """A recovered item is valid if it was found on/after the lost date
        and within MAX_DAYS_GAP days. Rows with no date are kept."""
        gap = (self.recovered["date"] - query["date"]).dt.days
        valid = gap.between(0, MAX_DAYS_GAP) | gap.isna()
        return valid.to_numpy(), gap.to_numpy()

    @staticmethod
    def date_score(gap):
        """1.0 if found the same day, falling to 0 at MAX_DAYS_GAP."""
        if pd.isna(gap):
            return 0.5
        return max(0.0, 1 - gap / MAX_DAYS_GAP)

    # -----------------------------------------------------------------------
    # 4. Calculate similarity + 6. rank and return top k
    # -----------------------------------------------------------------------
    def match(self, query, k=TOP_K, method="cosine"):
        """Rank recovered items for one query (a dict or a pandas row).

        method="cosine" -> primary algorithm (cosine similarity)
        method="knn"    -> comparison algorithm (KNN, Euclidean distance)
        """
        q = pd.DataFrame([query]).copy()
        for col in CATEGORICAL_FIELDS + ["description"]:
            q[col] = q[col].apply(clean_text)
        q["category"] = q["category"].replace(CATEGORY_MAP)
        q["date"] = pd.to_datetime(q["date"], errors="coerce")
        q_vec = self.vectorize(q)
        q_row = q.iloc[0]

        valid, gaps = self.valid_candidates(q_row)

        # Also invalid: candidates that share no field or word at all.
        overlap = (q_vec @ self.recovered_X.T).toarray()[0] > 0
        valid = valid & overlap

        if method == "cosine":
            field_sim = cosine_similarity(q_vec, self.recovered_X)[0]
        elif method == "knn":
            n = self.recovered_X.shape[0]
            knn = NearestNeighbors(n_neighbors=n, metric="euclidean")
            knn.fit(self.recovered_X)
            dist, idx = knn.kneighbors(q_vec)
            # Convert distance to a 0-1 similarity (closer = higher).
            max_dist = np.sqrt(2 * sum(FIELD_WEIGHTS.values()))
            field_sim = np.zeros(n)
            field_sim[idx[0]] = 1 - dist[0] / max_dist
        else:
            raise ValueError("method must be 'cosine' or 'knn'")

        results = []
        for i in np.where(valid)[0]:
            score = (1 - DATE_WEIGHT) * field_sim[i] + DATE_WEIGHT * self.date_score(
                gaps[i]
            )
            cand = self.recovered.iloc[i]
            results.append(
                {
                    "item_id": cand["item_id"],
                    "category": cand["category"],
                    "color": cand["color"],
                    "brand": cand["brand"],
                    "location": cand["location"],
                    "date": cand["date"].date() if pd.notna(cand["date"]) else None,
                    "description": cand["description"],
                    "score": round(float(score), 3),
                    "explanation": self.explain(q_row, cand, gaps[i]),
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        top = results[:k]
        strong = bool(top) and top[0]["score"] >= MIN_SCORE
        return top, strong

    def explain(self, q, cand, gap):
        """Plain-language reason for a match."""
        same = [
            f
            for f in CATEGORICAL_FIELDS
            if q[f] == cand[f] and q[f] not in UNKNOWN_VALUES
        ]
        analyzer = self.tfidf.build_analyzer()
        shared_words = sorted(
            set(analyzer(q["description"])) & set(analyzer(cand["description"]))
        )
        parts = []
        if same:
            parts.append("same " + ", ".join(same))
        if shared_words:
            parts.append("shared words: " + ", ".join(shared_words[:5]))
        if not pd.isna(gap):
            parts.append(f"found {int(gap)} day(s) after lost")
        return "; ".join(parts) if parts else "weak overlap"


# ---------------------------------------------------------------------------
# Evaluation (uses matched_id as ground truth; not used for matching)
# ---------------------------------------------------------------------------
def evaluate(matcher, method):
    rows = []
    for _, item in matcher.reported.iterrows():
        top, strong = matcher.match(item, k=TOP_K, method=method)
        ids = [r["item_id"] for r in top]
        truth = item["matched_id"] if isinstance(item["matched_id"], str) else None
        rank = ids.index(truth) + 1 if truth in ids else None
        rows.append(
            {
                "reported_id": item["item_id"],
                "true_match": truth or "-",
                "top1": ids[0] if ids else "-",
                "top1_score": top[0]["score"] if top else 0.0,
                "rank_of_true": rank,
                "strong_match": strong,
            }
        )
    res = pd.DataFrame(rows)

    has_truth = res[res["true_match"] != "-"]
    no_truth = res[res["true_match"] == "-"]
    summary = {
        "method": method,
        "top1_accuracy": (has_truth["rank_of_true"] == 1).mean(),
        "top3_accuracy": (has_truth["rank_of_true"] <= 3).mean(),
        "top5_accuracy": has_truth["rank_of_true"].notna().mean(),
        "mrr": has_truth["rank_of_true"]
        .apply(lambda r: 1 / r if pd.notna(r) else 0)
        .mean(),
        "avg_top1_score_true_pairs": has_truth["top1_score"].mean(),
        "avg_top1_score_no_match": no_truth["top1_score"].mean(),
        "no_match_correctly_flagged": (~no_truth["strong_match"]).mean(),
        "coverage": res["strong_match"].mean(),
    }
    return res, summary


def print_matches(title, top, strong):
    print(f"\n{title}")
    if not strong:
        print(
            "  No strong match found. Ask the owner for more details or check again later."
        )
    for rank, r in enumerate(top, 1):
        print(
            f"  {rank}. {r['item_id']}  score={r['score']:.3f}  "
            f"{r['color']} {r['brand']} {r['category']} @ {r['location']} ({r['date']})"
        )
        print(f"     why: {r['explanation']}")


if __name__ == "__main__":
    df, report = load_and_clean()
    print("=== Cleaning report ===")
    for key, val in report.items():
        print(f"  {key}: {val}")

    matcher = LostFoundMatcher(df)
    print(
        f"\nReported items: {len(matcher.reported)}  |  Recovered items: {len(matcher.recovered)}"
    )

    # Example 1: a record from the dataset (normal case)
    example = matcher.reported[matcher.reported["item_id"] == "R001"].iloc[0]
    top, strong = matcher.match(example, k=3)
    print_matches("=== Top 3 for R001 (cosine) ===", top, strong)

    # Example 2: a new report typed by a user (what Streamlit will send)
    new_report = {
        "category": "Phone",
        "color": "Blue",
        "brand": "Samsung",
        "location": "Gymnasium",
        "date": "2026-09-19",
        "description": "samsung phone with flower case",
    }
    top, strong = matcher.match(new_report, k=3)
    print_matches("=== Top 3 for a new report (cosine) ===", top, strong)

    # Example 3: an item with no true match
    example = matcher.reported[matcher.reported["item_id"] == "R028"].iloc[0]
    top, strong = matcher.match(example, k=3)
    print_matches(
        "=== Top 3 for R028, yellow raincoat (no true match) ===", top, strong
    )

    # Example 4: incomplete input (blank description, unknown brand)
    incomplete = {
        "category": "keys",
        "color": "silver",
        "brand": "",
        "location": "Parking Lot",
        "date": "2026-09-20",
        "description": "",
    }
    top, strong = matcher.match(incomplete, k=3)
    print_matches("=== Top 3 for incomplete input (no description) ===", top, strong)

    # Compare cosine vs KNN
    all_results, summaries = [], []
    for method in ["cosine", "knn"]:
        res, summary = evaluate(matcher, method)
        res.insert(0, "method", method)
        all_results.append(res)
        summaries.append(summary)

    comparison = pd.DataFrame(summaries).set_index("method").round(3)
    print("\n=== Cosine vs KNN comparison ===")
    print(comparison.T.to_string())

    pd.concat(all_results).to_csv("match_results.csv", index=False)
    comparison.to_csv("comparison_table.csv")
    print("\nSaved match_results.csv and comparison_table.csv")
