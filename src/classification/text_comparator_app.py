import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def app():
    st.title("Text Comparator")

    model = load_model()

    st.header("Upload CSV or XLSX Files")
    uploaded_file_1 = st.file_uploader("Upload the first file (CSV or XLSX)", type=["csv", "xlsx"])
    uploaded_file_2 = st.file_uploader("Upload the second file (CSV or XLSX)", type=["csv", "xlsx"])

    def read_file(uploaded_file):
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
                return pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read XLSX file: {e}")
                return None
        else:
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
            return None

    if uploaded_file_1 and uploaded_file_2:
        try:
            df1 = read_file(uploaded_file_1)
            df2 = read_file(uploaded_file_2)
            if df1 is None or df2 is None:
                return

            st.header("Select Columns to Compare")
            col1 = st.selectbox("Select column from the first file", df1.columns)
            col2 = st.selectbox("Select column from the second file to compare", df2.columns)
            output_col = st.selectbox("Select column from the second file to output", df2.columns)

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
                        
                        if best_match_score.item() >= similarity_threshold:
                            best_match_idx_item = best_match_idx.item()
                            similar_text = texts2[best_match_idx_item]
                            output_value = df2.loc[best_match_idx_item, output_col]
                            if pd.isna(output_value) or str(output_value).strip() == "":
                                output_value = blank_output_fill
                            results.append({
                                "Text from File 1": texts1[i],
                                "Most Similar Text from File 2": similar_text,
                                "Similarity Score": f"{best_match_score.item():.4f}",
                                f"Output from '{output_col}'": output_value
                            })
                        else:
                            results.append({
                                "Text from File 1": texts1[i],
                                "Most Similar Text from File 2": no_similar_text_fill,
                                "Similarity Score": f"{best_match_score.item():.4f}",
                                f"Output from '{output_col}'": no_similar_text_fill
                            })
                
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
