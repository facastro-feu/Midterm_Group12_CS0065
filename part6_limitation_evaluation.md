# Part 6: Testing, Comparison, and Reflection (Group 12)

## 1. Test cases

| ID | Case | Input | Expected | Actual | Result |
|---|---|---|---|---|---|
| N1 | Normal | R001: black Jansport backpack, Library, 2026-09-14 | F001 ranked #1, Strong | F001 #1, 0.837 Strong (KNN 0.629) | Pass |
| N2 | Normal | New report: blue Samsung phone, Gymnasium, 2026-09-19, "samsung phone with flower case" | F005 ranked #1 | F005 #1, 0.930 Strong (KNN 0.760) | Pass |
| N3 | Normal | R022: gray Casio calculator, Engineering Building | F022 ranked #1 | F022 #1, 0.913 Strong; next best only 0.255 | Pass |
| M1 | Missing | Form submitted with every field blank | Rejected with clear errors | 4 errors (category, color, location, date) + notes about brand and description | Pass |
| M2 | Missing | Upload row L3: no category (red umbrella, Library) | Row rejected | "Category is required", row skipped | Pass |
| M3 | Missing | R023: white Apple charger, blank description | F023 still found from the other fields | F023 #1, 0.870 Strong; app notes "No description given" | Pass |
| M4 | Missing | Silver keys, Parking Lot, 2026-09-20, no brand, no description | F006 ranked #1 | F006 #1, 0.848 Strong; next best 0.412 | Pass |
| M5 | Missing | R002: brand Nike, but F002 has brand unknown | F002 still ranked #1 | F002 #1, 0.792 Strong | Pass |
| B1 | Boundary | Black North Face jacket lost 2026-09-12; F018 found 2026-10-12 (exactly 30 days) | F018 kept (30 days is allowed) | F018 #1, 0.670 Possible (date adds nothing at 30 days) | Pass |
| B2 | Boundary | Same jacket lost 2026-09-11 (31 days) | F018 removed | F018 removed; #1 is F036 (a different black hoodie), 0.476 Possible | Pass |
| B3 | Boundary | Black Jansport backpack lost 2026-09-16; F001 and F031 were found on 09-15 | Items found before the loss date removed | Both removed; #1 is F002, 0.608 Possible | Pass |
| B4 | Boundary | Upload row with date 2026-10-20 (future) | Row rejected | "Date cannot be in the future." (the form's date picker blocks future dates) | Pass |
| B5 | Boundary | Upload row L2: date 19/09/2026 | Row rejected | "Date is missing or invalid. Use the format YYYY-MM-DD." | Pass |
| B6 | Boundary | Brown wallet, Cafeteria, 2026-09-17, description just "wallet" | Warning, but still ranked | Short-description warning; F003 0.802, F028 0.682 (two brown wallets close together) | Pass |
| B7 | Boundary | Upload row with category "skateboard" (not in our records) | Warning and no match | Warning says it "cannot match any found item", yet #1 is F027 (a phone), 0.466 Possible | **Fail** |
| U1 | Uncertain | R013: owner said gray laptop, found one is black | F013 #1, lower confidence | F013 #1, 0.666 Possible (KNN 0.466) | Pass |
| U2 | Uncertain | R028: yellow raincoat (no true match) | "No strong match found" | Warning shown; best F018, 0.379 Weak | Pass |
| U3 | Uncertain | R025: "red phone case only" (no true match) | "No strong match found" | F027 0.661 Possible (same category, location, words "phone", "case") | **Fail** |
| U4 | Uncertain | R011: vague "water bottle" | F011 #1 | F035 (look-alike) 0.865, F011 #2 at 0.814 | Partial (top 3) |
| U5 | Uncertain | R024: vague "lost my wallet" | F024 #1 | F034 (look-alike) 0.811, F024 #2 at 0.727 | Partial (top 3) |
| U6 | Uncertain | R017: owner said black hoodie, found one is gray | F017 #1 | F036 (black hoodie) 0.868, F017 #2 at 0.732 | Partial (top 3) |

**Summary:** 16 pass, 3 partial, 2 fail out of 21. Every true match landed in the top 3. Validation (missing fields, bad and future dates) worked in every case.

## 2. Cosine vs KNN

| Measure | Cosine | KNN |
|---|---|---|
| Top-1 accuracy | 87.5% (21/24) | 87.5% (21/24) |
| Top-3 accuracy | 100% | 100% |
| Mean reciprocal rank (MRR) | 0.938 | 0.938 |
| Avg #1 score, items with a true match | 0.842 | 0.658 |
| Avg #1 score, items with no true match | 0.362 | 0.300 |
| Gap between the two averages | **0.48** | 0.36 |
| Lowest #1 score among true matches | 0.666 (R013) | 0.466 (R013) |
| True matches labeled Strong (≥ 0.70) | 23/24 | 7/24 |
| No-match items correctly flagged (< 0.40) | 5/6 | 5/6 |

![Cosine vs KNN: top-1 scores](cosine_vs_knn_chart.png)

**Interpretation:** both methods use the same weighted features, so they put items in the same order and get the same accuracy. The difference is the score. Cosine gives real matches high scores and non-matches low ones, with a wide gap between them, so one cutoff (0.40) and one "Strong" line (0.70) work well. KNN squeezes everything closer together: its weakest real match (0.466) sits just above the cutoff, and only 7 of 24 real matches reach "Strong". KNN would need its own cutoffs. That is why cosine is the better primary method here: same ranking, but scores that are easier to explain to staff.

## 3. Critical-thinking questions

**Q1. Which attributes should contribute most?**

Category and description (weight 3 each), then color (2), then brand and location (1.5 each), with the date as a small tiebreaker (15% of the score).

- **Category** goes first because a wallet can never be a phone.
- **Description** holds the details that tell look-alikes apart (keychain, stickers, cracked screen).
- **Color** helps a lot, but owners sometimes get it wrong (R005, R013, R017), so it should not outweigh the others.
- **Brand** is often unknown or only known on one side (F002, F012).
- **Location** is useful but weak, because items get turned in somewhere else (F003, F010, F018).
- **Date** mostly separates two similar items; anything found before the loss or more than 30 days later is removed outright.

**Q2. What should the system show when there is no strong match?**

If the best score is below 0.40, say **"No strong match found"** first, then still list the closest items labeled **Weak**, with the reason each one appeared. Then suggest a next step: ask the owner for more detail (marks, stickers, contents, brand) and check again in a few days, since the item may not have been turned in yet. Our app does this (test U2).

**Q3. How do we keep results understandable and avoid repeated or unfair results?**

- **Understandable:** show a plain label (Strong / Possible / Weak) next to every score, plus a one-line reason ("same category, color, brand; shared words: backpack, keychain; found 1 day after lost").
- **Repeated:** we remove duplicate records (R031 was a copy of R010). The same found item can still show up for several reports: F018 and F020 are each the #1 suggestion for 3 reported items, usually as a Weak fallback. Labeling those as Weak keeps them from looking like real matches. Marking an item as returned and removing it from the pool would also help (not built).
- **Unfair:** detailed reports score higher than vague ones (R011 and R024 lost to look-alikes). Someone who struggles to describe an item in English is at a disadvantage. Dropdowns for category, color, and location reduce this, and staff should ask follow-up questions rather than trust the #1 result.
- **Safety:** the system only suggests. Staff confirm ownership with a detail that is not in the listing before releasing anything, which also stops false claims.

## 4. Interpretation and limitations

**Interpretation:** weighted cosine similarity found the right item first for 21 of 24 reports and in the top 3 for all 24. It flagged 5 of 6 no-match reports correctly. KNN ranks the same way, but its scores are harder to put a cutoff on. Most mistakes come from vague or wrong reports competing with look-alike items, which is the same thing a human would struggle with.

**Limitations:**

- **Small, made-up dataset.** 66 records written by our group, with the true pairs also decided by us. Real reports would be messier.
- **Tuned on the same data we tested on.** The weights, the 0.40 and 0.70 cutoffs, and the 30-day limit were picked by judgment while looking at this data. There is no separate test set, so real accuracy is probably lower.
- **Missing fields can inflate cosine scores.** Cosine only compares the fields that are filled in, so a report with an unknown category is judged on color and location alone (test B7).
- **Exact words only.** TF-IDF does not know that "cardholder" means "wallet" or that "dark" is close to "black". Our synonym list covers only a few categories.
- **No partial credit for colors.** Gray vs black counts as completely different, so one mistake by the owner costs a lot (R013, R017).
- **No photos.** Many lost-and-found offices log pictures, which we do not use.
- **KNIME uses fewer fields.** The KNIME workflow compares category, color, brand, and location only (no description or date), so its scores differ from the Python version.
- **Cutoffs fit cosine only.** The Strong/Possible/Weak labels were set for cosine; under KNN, most real matches show as Possible.
