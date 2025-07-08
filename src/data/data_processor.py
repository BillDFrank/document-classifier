import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def generate_label_sequence(group):
    """Generate a label sequence with counts for each envelope."""
    label_counts = group['label'].value_counts().sort_index()
    return '->'.join([f"{label}:{count}" for label, count in label_counts.items()])

def aggregate_embeddings(embeddings):
    """Aggregate embeddings by summing them."""
    return np.sum(np.stack(embeddings), axis=0)

def app():
    st.title('Process Comparison')
    st.write("Identifies similar processes based on a reference process.")

    if 'df' not in st.session_state:
        st.warning("Nenhum dado carregado. Por favor, carregue um arquivo Parquet na seção 'Datasource'.")
        return

    df = st.session_state.df

    # Input for the reference envelope
    reference_envelope = st.sidebar.text_input("Enter the reference envelope number")

    if reference_envelope:
        # Ensure 'label' column contains strings and filter invalid entries
        df['label'] = df['label'].astype(str)
        df_filtered = df[df['label'].notna() & (df['label'] != "") & (df['label'] != "nan")]

        if df_filtered.empty:
            st.error("No documents with valid labels found.")
            return

        st.session_state.df = df_filtered

        # Group documents by envelope and create a label sequence with counts
        df_grouped = df_filtered.groupby('envelope').apply(generate_label_sequence).reset_index()
        df_grouped.columns = ['envelope', 'label_sequence']

        # Identify the label sequence of the reference envelope
        reference_label_sequence = df_grouped[df_grouped['envelope'] == reference_envelope]['label_sequence'].values

        if len(reference_label_sequence) == 0:
            st.warning("Reference envelope not found or has no valid labels.")
            return

        reference_label_sequence = reference_label_sequence[0]
        st.write(f"Label sequence of the reference envelope: {reference_label_sequence}")

        # Find processes with the same label sequence
        common_envelopes = df_grouped[df_grouped['label_sequence'] == reference_label_sequence]['envelope']
        df_same_sequence = df_filtered[df_filtered['envelope'].isin(common_envelopes)]

        # Aggregate embeddings per process
        df_embeddings = df_same_sequence.groupby('envelope')['embedding_completo'].apply(lambda x: aggregate_embeddings(x.tolist())).reset_index()

        # Calculate similarity using cosine similarity
        similarity_results = []
        reference_embedding = df_embeddings[df_embeddings['envelope'] == reference_envelope]['embedding_completo'].values

        if len(reference_embedding) == 0:
            st.error("No embedding found for the reference envelope.")
            return

        reference_embedding = reference_embedding[0]

        for i in range(len(df_embeddings)):
            envelope = df_embeddings.iloc[i]['envelope']
            if envelope == reference_envelope:
                continue

            sim = cosine_similarity([reference_embedding], [df_embeddings.iloc[i]['embedding_completo']])[0][0]
            similarity_results.append({
                'envelope': envelope,
                'similarity': sim
            })

        # Check if similarity results are available
        if not similarity_results:
            st.warning("No similar processes found.")
            return

        # Convert results to DataFrame and find the most similar process
        df_similarity = pd.DataFrame(similarity_results)
        most_similar_process = df_similarity.loc[df_similarity['similarity'].idxmax()]

        st.write(f"Most similar process to envelope {reference_envelope}: {most_similar_process['envelope']} with similarity of {most_similar_process['similarity']:.4f}")

        # Display 'ds_documento_ocr' content side by side with labels
        proc1_docs = df_filtered[df_filtered['envelope'] == reference_envelope].sort_values(by='label')[['label', 'ds_documento_ocr']].values.tolist()
        proc2_docs = df_filtered[df_filtered['envelope'] == most_similar_process['envelope']].sort_values(by='label')[['label', 'ds_documento_ocr']].values.tolist()

        # Ensure documents are displayed side by side, even if the number of documents differs
        max_docs = max(len(proc1_docs), len(proc2_docs))
        proc1_docs.extend([["", ""]] * (max_docs - len(proc1_docs)))  # Pad with empty strings
        proc2_docs.extend([["", ""]] * (max_docs - len(proc2_docs)))

        for (label1, doc1), (label2, doc2) in zip(proc1_docs, proc2_docs):
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Label: {label1}")
                st.text_area("Reference Process", doc1[:1000], height=150)
            with col2:
                st.text(f"Label: {label2}")
                st.text_area("Similar Process", doc2[:1000], height=150)

if __name__ == "__main__":
    app()
