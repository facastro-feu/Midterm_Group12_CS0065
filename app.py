import datetime as dt

import pandas as pd
import streamlit as st

# Part 4 code - the matching engine
from lost_found_matcher import (
    load_and_clean,
    LostFoundMatcher,
    MIN_SCORE,
    CATEGORY_MAP,
    clean_text,
)

REQUIRED_FIELDS = ["category", "color", "brand", "location", "date", "description"]

st.set_page_config(page_title="Lost and Found Matcher", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Load the dataset and the matcher once (Streamlit reruns the file on each click)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_matcher():
    df, report = load_and_clean()
    return LostFoundMatcher(df), report


matcher, clean_report = get_matcher()
known = matcher.df  # cleaned dataset, used to fill the dropdowns


def options(field):
    values = sorted(v for v in known[field].unique() if v and v != "unknown")
    return values


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(item):
    """Return (errors, warnings) for one lost-item report."""
    errors, warnings = [], []

    for field in ["category", "color", "location"]:
        if not clean_text(item.get(field)):
            errors.append(f"**{field.capitalize()}** is required.")

    category = CATEGORY_MAP.get(
        clean_text(item.get("category")), clean_text(item.get("category"))
    )
    if category and category not in options("category"):
        warnings.append(
            f"Category **{category}** is not in our records, so it cannot match any found item."
        )

    raw_date = item.get("date")
    if isinstance(raw_date, str):
        # typed or uploaded text must be exactly YYYY-MM-DD
        date = pd.to_datetime(raw_date.strip(), format="%Y-%m-%d", errors="coerce")
    else:
        date = pd.to_datetime(raw_date, errors="coerce")  # from the date picker
    if pd.isna(date):
        errors.append("**Date** is missing or invalid. Use the format YYYY-MM-DD.")
    elif date.date() > dt.date.today():
        errors.append("**Date** cannot be in the future.")

    if not clean_text(item.get("brand")):
        warnings.append("No brand given, so brand will not be used in matching.")
    if not clean_text(item.get("description")):
        warnings.append(
            "No description given. Matching will rely on the other fields only."
        )
    elif len(clean_text(item.get("description")).split()) < 3:
        warnings.append(
            "The description is very short. Adding details (marks, stickers, contents) improves matching."
        )

    return errors, warnings


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
def show_results(top, strong, k):
    if not top:
        st.error(
            "No possible matches. No recovered item shares any detail with this report "
            "within the valid date range."
        )
        return
    if not strong:
        st.warning(
            f"**No strong match found.** The best score is below {MIN_SCORE:.2f}. "
            "Showing the closest items anyway. Ask the owner for more details or check again later."
        )
    else:
        st.success(
            f"Found {len(top)} possible match(es). Best score: **{top[0]['score']:.3f}**"
        )

    table = pd.DataFrame(top)[
        [
            "item_id",
            "score",
            "category",
            "color",
            "brand",
            "location",
            "date",
            "description",
            "explanation",
        ]
    ]
    table.insert(0, "rank", range(1, len(table) + 1))
    table["strength"] = table["score"].apply(
        lambda s: "Strong" if s >= 0.70 else ("Possible" if s >= MIN_SCORE else "Weak")
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "score": st.column_config.ProgressColumn(
                "similarity score", min_value=0, max_value=1, format="%.3f"
            ),
            "item_id": "found item",
            "explanation": st.column_config.TextColumn("why it matched", width="large"),
        },
    )
    st.caption(
        "Scores go from 0 (nothing in common) to 1 (identical). "
        "Staff should always verify ownership before releasing an item."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Lost and Found Item Matcher")
st.write("Enter a lost-item report to find the most similar recovered items.")

with st.sidebar:
    st.header("Settings")
    k = st.radio("Number of matches to show", [3, 5], index=1, horizontal=True)
    method_label = st.radio(
        "Algorithm", ["Cosine similarity (primary)", "K-Nearest Neighbors (comparison)"]
    )
    method = "cosine" if method_label.startswith("Cosine") else "knn"
    st.divider()
    st.caption(
        f"Dataset: {clean_report['clean_rows']} records "
        f"({len(matcher.reported)} reported, {len(matcher.recovered)} recovered). "
        f"Removed duplicates: {', '.join(clean_report['duplicates_removed']) or 'none'}."
    )

tab_form, tab_upload = st.tabs(["Enter a report", "Upload reports (CSV)"])

# --- Option 1: type one report ---------------------------------------------
with tab_form:
    with st.form("report"):
        c1, c2, c3 = st.columns(3)
        category = c1.selectbox("Item category *", [""] + options("category"))
        color = c2.selectbox("Color *", [""] + options("color"))
        brand = c3.text_input(
            "Brand", placeholder="e.g. Jansport (leave blank if unknown)"
        )
        c4, c5 = st.columns(2)
        location = c4.selectbox("Location lost *", [""] + options("location"))
        date = c5.date_input("Date lost *", value=None, max_value=dt.date.today())
        description = st.text_area(
            "Description",
            placeholder="e.g. black backpack with a small keychain on the zipper",
        )
        submitted = st.form_submit_button("Find matches", type="primary")

    if submitted:
        item = {
            "category": category,
            "color": color,
            "brand": brand,
            "location": location,
            "date": date,
            "description": description,
        }
        errors, warnings = validate(item)
        for e in errors:
            st.error(e)
        for w in warnings:
            st.info(w)
        if not errors:
            top, strong = matcher.match(item, k=k, method=method)
            show_results(top, strong, k)

# --- Option 2: upload a CSV of reports -------------------------------------
with tab_upload:
    st.write(
        "Upload a CSV with these columns: "
        + ", ".join(f"`{c}`" for c in REQUIRED_FIELDS)
    )
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        try:
            reports = pd.read_csv(file, dtype=str, keep_default_na=False)
        except Exception as exc:  # unreadable file
            st.error(f"Could not read the file: {exc}")
            reports = None

        if reports is not None:
            missing_cols = [c for c in REQUIRED_FIELDS if c not in reports.columns]
            if missing_cols:
                st.error("Missing column(s): " + ", ".join(missing_cols))
            elif reports.empty:
                st.error("The file has no rows.")
            else:
                st.write(f"Loaded **{len(reports)}** report(s).")
                for i, row in reports.iterrows():
                    label = row.get("item_id") or f"Row {i + 1}"
                    with st.expander(
                        f"{label}: {row['color']} {row['category']} at {row['location']}"
                    ):
                        item = {c: row[c] for c in REQUIRED_FIELDS}
                        errors, warnings = validate(item)
                        for e in errors:
                            st.error(e)
                        for w in warnings:
                            st.info(w)
                        if errors:
                            st.write("Skipped because of the errors above.")
                            continue
                        top, strong = matcher.match(item, k=k, method=method)
                        show_results(top, strong, k)
