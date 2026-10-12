# Data Dictionary: Lost and Found Item Matching (Group 12)

Dataset file: `lost_found_dataset.csv`
Raw size: 67 rows (31 reported, 36 recovered). After removing 1 duplicate: 66 records.

## Columns

| Column | Type | Role | Used in similarity? | Description | Allowed values / format |
|---|---|---|---|---|---|
| item_id | text | Reference-only identifier | No | Unique ID. `R###` = reported item, `F###` = recovered (found) item. | R001-R031, F001-F036 |
| record_type | categorical | Reference / filter | No (used to split the two piles) | Whether the item was reported lost or recovered by staff. | `reported`, `recovered` |
| category | categorical | Analytical input | Yes (high weight) | Type of item. | backpack, wallet, phone, keys, umbrella, water bottle, laptop, earphones, id card, jacket, book, eyeglasses, watch, calculator, charger |
| color | categorical | Analytical input | Yes (medium weight) | Main color of the item. | black, blue, brown, gray, green, gold, red, silver, white, yellow |
| brand | categorical | Analytical input | Yes (medium weight) | Brand name; `unknown` when not visible or not known. | e.g. Apple, Jansport, Casio, unknown |
| location | categorical | Analytical input | Yes (medium weight) | Campus place where the item was lost or found. | Library, Cafeteria, Gymnasium, Main Gate, Parking Lot, Student Center, Computer Lab, Bus Stop, Auditorium, Engineering Building |
| date | date | Analytical input | Yes (low weight, as date gap in days) | Date lost (reported) or found (recovered). | YYYY-MM-DD |
| description | free text | Analytical input | Yes (high weight, TF-IDF) | Short description of distinguishing features. May be blank. | Lowercase text, no personal data |
| matched_id | text | Ground-truth label (evaluation only) | No | The true counterpart record, used only to measure top-k accuracy in Part 6. Blank means no true match exists. | An `item_id` or blank |

## Part 1 Classification

- **Reference-only fields:** `item_id`, `record_type`
- **Analytical inputs:** `category`, `color`, `brand`, `location`, `date`, `description`
- **Required target label:** None for the algorithm, since similarity search is unsupervised. `matched_id` is kept only to evaluate results.
- **Generated output:** Ranked list of possible matches with similarity scores
- **Missing information:** Blank descriptions (R023, R027), `unknown` brands (including one-sided ones like R002 Nike vs F002 unknown), vague descriptions (R011 "water bottle", R024 "lost my wallet", F028, F030), color disagreements between pairs (R005, R013, R017), and items turned in at a different place than where they were lost (F003, F010, F018)
- **Concern (privacy and safety):** Descriptions and ID cards can contain names, phone numbers, or other personal data. The system should only suggest possible matches; staff must verify ownership (for example by asking the claimant for an unlisted detail) before releasing an item, which also guards against false claims.

## Planned cleaning steps (Part 2)

1. Strip whitespace and lowercase all text (fixes ` Wallet `, `Laptop`, `UNKNOWN`, `north face`, `  RED umbrella`).
2. Standardize categories (`headphones` -> `earphones`).
3. Standardize brand case (`north face` and `North Face` -> `north face`).
4. Remove duplicates by content (R031 duplicates R010).
5. Flag blank descriptions (R023, R027) and fill with an empty string so TF-IDF still runs.
6. Keep `item_id` untouched.

## Deliberate test cases built into the data

| Case type | Examples |
|---|---|
| Normal (clear true pair) | R001/F001, R004/F004, R022/F022 |
| Missing description | R023, R027 |
| Missing brand on both sides (`unknown`) | R006, R008, R016 |
| Brand known on only one side | R002/F002, R012/F012, R021/F021 |
| Color conflict | R005 (black) vs F005 (blue), R013 (gray) vs F013 (black), R017 (black) vs F017 (gray) |
| Different location | R003/F003, R010/F010, R018/F018 |
| Long date gap | R007/F007 (7 days), R018/F018 (10 days), R020/F020 (12 days) |
| Vague report | R011 ("water bottle"), R024 ("lost my wallet") |
| Look-alike distractors (no true match) | F031 (black backpack), F032 (white earbuds), F033 (black iPhone), F034 (black wallet), F035 (green bottle), F036 (black hoodie) |
| No true match | R025-R030 (reported), F025-F036 (recovered) |
| Duplicate to remove | R031 |

These cases are there on purpose so the results are realistic (about 87.5% top-1 accuracy) rather than a perfect 100%.
