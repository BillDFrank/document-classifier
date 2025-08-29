import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_excel_sheet_names(uploaded_file):
    """Get sheet names from Excel file without reading the entire file"""
    if uploaded_file is None:
        return []
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        return excel_file.sheet_names
    except Exception as e:
        st.error(f"Failed to read Excel file sheets: {e}")
        return []

def app():
    st.title("Text Comparator")

    model = load_model()

    # Initialize session state for sheet selections
    if 'sheet_1_selected' not in st.session_state:
        st.session_state.sheet_1_selected = None
    if 'sheet_2_selected' not in st.session_state:
        st.session_state.sheet_2_selected = None
    if 'file_1_processed' not in st.session_state:
        st.session_state.file_1_processed = False
    if 'file_2_processed' not in st.session_state:
        st.session_state.file_2_processed = False

    st.header("Upload CSV or XLSX Files")
    uploaded_file_1 = st.file_uploader("Upload the first file (CSV or XLSX)", type=["csv", "xlsx"])
    uploaded_file_2 = st.file_uploader("Upload the second file (CSV or XLSX)", type=["csv", "xlsx"])

    def read_file(uploaded_file, sheet_name=None):
        if uploaded_file is None:
            return None
        if uploaded_file.name.lower().endswith('.csv'):
            try:
                return pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding='latin1')
                except Exception as e:
                    st.error(f"Failed to read CSV file: {e}")
                    return None
            except Exception as e:
                st.error(f"Failed to read CSV file: {e}")
                return None
        elif uploaded_file.name.lower().endswith('.xlsx'):
            try:
                if sheet_name is not None:
                    return pd.read_excel(uploaded_file, sheet_name=sheet_name)
                else:
                    return pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read XLSX file: {e}")
                return None
        else:
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
            return None

    # Process first file
    if uploaded_file_1:
        if uploaded_file_1.name.lower().endswith('.xlsx'):
            sheet_names_1 = get_excel_sheet_names(uploaded_file_1)
            if sheet_names_1:
                selected_sheet_1 = st.selectbox(
                    "Select sheet from first Excel file",
                    sheet_names_1,
                    key="sheet_1_selector"
                )
                st.session_state.sheet_1_selected = selected_sheet_1
                st.session_state.file_1_processed = True
            else:
                st.error("Could not read sheets from first Excel file.")
                st.session_state.file_1_processed = False
        else:
            # CSV file - no sheet selection needed
            st.session_state.file_1_processed = True
            st.session_state.sheet_1_selected = None

    # Process second file
    if uploaded_file_2:
        if uploaded_file_2.name.lower().endswith('.xlsx'):
            sheet_names_2 = get_excel_sheet_names(uploaded_file_2)
            if sheet_names_2:
                selected_sheet_2 = st.selectbox(
                    "Select sheet from second Excel file",
                    sheet_names_2,
                    key="sheet_2_selector"
                )
                st.session_state.sheet_2_selected = selected_sheet_2
                st.session_state.file_2_processed = True
            else:
                st.error("Could not read sheets from second Excel file.")
                st.session_state.file_2_processed = False
        else:
            # CSV file - no sheet selection needed
            st.session_state.file_2_processed = True
            st.session_state.sheet_2_selected = None

    # Only proceed if both files are uploaded and processed
    if uploaded_file_1 and uploaded_file_2 and st.session_state.file_1_processed and st.session_state.file_2_processed:
        try:
            # Read files with selected sheets
            df1 = read_file(uploaded_file_1, st.session_state.sheet_1_selected)
            df2 = read_file(uploaded_file_2, st.session_state.sheet_2_selected)

            if df1 is None or df2 is None:
                return

            st.header("Select Columns to Compare")
            col1 = st.selectbox("Select column from the first file", df1.columns)
            col2 = st.selectbox("Select column from the second file to compare", df2.columns)

            st.header("Select Additional Columns to Output (Optional)")
            st.write("Choose columns from both files to include in the results:")

            # Multiselect for df1 columns (excluding the comparison column)
            available_df1_cols = [col for col in df1.columns if col != col1]
            selected_df1_cols = st.multiselect(
                "Select columns from first file to include in output",
                available_df1_cols,
                default=[],
                help="These columns will be included as 'df1.column_name' in the results"
            )

            # Multiselect for df2 columns (excluding the comparison column)
            available_df2_cols = [col for col in df2.columns if col != col2]
            selected_df2_cols = st.multiselect(
                "Select columns from second file to include in output",
                available_df2_cols,
                default=[],
                help="These columns will be included as 'df2.column_name' in the results"
            )

            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)

            st.header("Custom Fill Options")
            no_similar_text_fill = st.text_input("Text to fill when no similar text is found", "No similar text found above threshold")
            blank_output_fill = st.text_input("Text to fill when output is blank", "-")

            if st.button("Compare"):
                with st.spinner("Comparing texts..."):
                    texts1 = df1[col1].astype(str).tolist()
                    texts2 = df2[col2].astype(str).tolist()

                    embeddings1 = model.encode(texts1, convert_to_tensor=True)
                    embeddings2 = model.encode(texts2, convert_to_tensor=True)

                    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

                    results = []
                    for i in range(len(texts1)):
                        best_match_score, best_match_idx = cosine_scores[i].max(dim=0)

                        # Create base result dictionary
                        result_dict = {
                            "Text from File 1": texts1[i],
                            "Similarity Score": f"{best_match_score.item():.4f}",
                        }

                        if best_match_score.item() >= similarity_threshold:
                            best_match_idx_item = best_match_idx.item()
                            similar_text = texts2[best_match_idx_item]
                            result_dict["Most Similar Text from File 2"] = similar_text

                            # Add selected columns from df1
                            for col in selected_df1_cols:
                                value = df1.loc[i, col]
                                if pd.isna(value):
                                    value = ""
                                result_dict[f"df1.{col}"] = str(value)

                            # Add selected columns from df2
                            for col in selected_df2_cols:
                                value = df2.loc[best_match_idx_item, col]
                                if pd.isna(value):
                                    value = ""
                                result_dict[f"df2.{col}"] = str(value)

                        else:
                            result_dict["Most Similar Text from File 2"] = no_similar_text_fill

                            # Add selected columns from df1 (no match found)
                            for col in selected_df1_cols:
                                value = df1.loc[i, col]
                                if pd.isna(value):
                                    value = ""
                                result_dict[f"df1.{col}"] = str(value)

                            # Add selected columns from df2 with fill value
                            for col in selected_df2_cols:
                                result_dict[f"df2.{col}"] = no_similar_text_fill

                        results.append(result_dict)

                st.header("Comparison Results")
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name='comparison_results.csv',
                        mime='text/csv',
                    )
                else:
                    st.info("No results to display.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
