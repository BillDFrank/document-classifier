import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def app():
    st.title("Text Comparator")

    model = load_model()

    st.header("Upload CSV Files")
    uploaded_file_1 = st.file_uploader("Upload the first CSV file", type="csv")
    uploaded_file_2 = st.file_uploader("Upload the second CSV file", type="csv")

    if uploaded_file_1 and uploaded_file_2:
        try:
            df1 = pd.read_csv(uploaded_file_1)
            df2 = pd.read_csv(uploaded_file_2)

            st.header("Select Columns to Compare")
            col1 = st.selectbox("Select column from the first file", df1.columns)
            col2 = st.selectbox("Select column from the second file to compare", df2.columns)
            output_col = st.selectbox("Select column from the second file to output", df2.columns)

            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)

            if st.button("Compare"):
                with st.spinner("Comparing texts..."):
                    # Get text from columns and convert to list
                    texts1 = df1[col1].astype(str).tolist()
                    texts2 = df2[col2].astype(str).tolist()

                    # Generate embeddings
                    embeddings1 = model.encode(texts1, convert_to_tensor=True)
                    embeddings2 = model.encode(texts2, convert_to_tensor=True)

                    # Compute cosine similarity
                    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

                    results = []
                    for i in range(len(texts1)):
                        # Find the best match for each text in file 1
                        best_match_score, best_match_idx = cosine_scores[i].max(dim=0)
                        
                        if best_match_score.item() >= similarity_threshold:
                            best_match_idx_item = best_match_idx.item()
                            similar_text = texts2[best_match_idx_item]
                            output_value = df2.loc[best_match_idx_item, output_col]
                            results.append({
                                "Text from File 1": texts1[i],
                                "Most Similar Text from File 2": similar_text,
                                "Similarity Score": f"{best_match_score.item():.4f}",
                                f"Output from '{output_col}'": output_value
                            })
                        else:
                            results.append({
                                "Text from File 1": texts1[i],
                                "Most Similar Text from File 2": "No similar text found above threshold",
                                "Similarity Score": f"{best_match_score.item():.4f}",
                                f"Output from '{output_col}'": "-"
                            })
                
                st.header("Comparison Results")
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                else:
                    st.info("No results to display.")


        except Exception as e:
            st.error(f"An error occurred: {e}")
