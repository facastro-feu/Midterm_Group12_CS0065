# Lost and Found Item Matching — Group 12 (CS0065 Midterm)

Prepared by: Castro, Calibjo, Valencia

- [Dataset](lost_found_dataset.csv)
- [Data dictionary](data_dictionary.md)
- [KNIME workflow](Group12_LostFound_Matching.knwf)
- [Python code](lost_found_matcher.py)
- [Streamlit app](app.py)
- [Evaluation report](part6_limitation_evaluation.md) / also available as [docx](part6_limitation_evaluation.docx)
- [Comparison chart](cosine_vs_knn_chart.png)
- [Screenshots](Screenshots/)

## Screenshots

**KNIME workflow**

<img src="Screenshots/KNIME_workflow.png" alt="KNIME workflow" width="100%">

**KNIME Top-k output**

<img src="Screenshots/KNIME_Top_K.png" alt="KNIME Top-k output" width="100%">

**Streamlit interface**

<img src="Screenshots/Streamlit_interface.png" alt="Streamlit interface" width="100%">

**Streamlit input**

<img src="Screenshots/Streamlit_input.png" alt="Streamlit input" width="100%">

**Streamlit output**

<img src="Screenshots/Streamlit_output.png" alt="Streamlit output" width="100%">

**Streamlit upload**

<img src="Screenshots/Streamlit_interface_upload.png" alt="Streamlit upload" width="100%">

## Results

| Measure                                  | Cosine (primary) | KNN (comparison) |
| ---------------------------------------- | :--------------: | :--------------: |
| Top-1 accuracy                           |      87.5%       |      87.5%       |
| Top-3 accuracy                           |       100%       |       100%       |
| Avg #1 score, reports with a true match  |      0.842       |      0.658       |
| Avg #1 score, reports with no true match |      0.362       |      0.300       |
| No-match reports correctly flagged       |      5 of 6      |      5 of 6      |

![Cosine vs KNN top-1 scores](cosine_vs_knn_chart.png)
