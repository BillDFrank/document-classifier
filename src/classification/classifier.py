import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# Directory where Parquet files are saved
PARQUET_DIR = os.path.join("data", "processed")
# Directory where models are saved
MODEL_DIR = "models"
# Directory to save prediction output
OUTPUT_DIR = "output"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def app():
    st.title('Classifier: Predict Labels for Unlabeled Data')

    # List available Parquet files
    parquet_files = [f for f in os.listdir(PARQUET_DIR) if f.endswith('.parquet')]
    if not parquet_files:
        st.error("No Parquet files found in data/processed directory.")
        return

    # List available models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and not f.endswith('_label_encoder.pkl')]
    if not model_files:
        st.error("No models found in models directory. Train a model first using Classifier Training.")
        return

    # Sidebar inputs
    selected_file = st.sidebar.selectbox("Select a Parquet file", parquet_files)
    selected_model_file = st.sidebar.selectbox("Select a Model", model_files)
    min_confidence = st.sidebar.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)

    # Button to start prediction
    if st.sidebar.button("Predict Labels"):
        # Load the Parquet file
        file_path = os.path.join(PARQUET_DIR, selected_file)
        df = pd.read_parquet(file_path)

        # Filter out rows with blank combined_text or filled labels
        df_unlabeled = df[
            (df['combined_text'].notnull()) & 
            (df['combined_text'].str.strip() != '') & 
            (df['label'].isnull() | (df['label'].str.strip() == ''))
        ]

        # Check for embedding column and drop rows with missing embeddings
        if 'embedding' not in df_unlabeled.columns:
            st.error("The Parquet file does not contain an 'embedding' column.")
            return
        df_unlabeled = df_unlabeled.dropna(subset=['embedding'])

        if df_unlabeled.empty:
            st.warning("No unlabeled data with embeddings found for prediction.")
            return

        st.write(f"Found {len(df_unlabeled)} unlabeled rows with embeddings for prediction.")

        # Load the model
        model_path = os.path.join(MODEL_DIR, selected_model_file)
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Load the corresponding LabelEncoder
        label_encoder_filename = f"{os.path.splitext(selected_model_file)[0]}_label_encoder.pkl"
        label_encoder_path = os.path.join(MODEL_DIR, label_encoder_filename)
        try:
            label_encoder = joblib.load(label_encoder_path)
        except Exception as e:
            st.error(f"Error loading LabelEncoder: {e}")
            return

        # Prepare features
        X_unlabeled = np.array(df_unlabeled['embedding'].tolist())

        # Get predictions and confidences
        if hasattr(model, "predict_proba"):
            # Get probabilities
            probabilities = model.predict_proba(X_unlabeled)
            confidences = np.max(probabilities, axis=1)  # Maximum probability as confidence
            y_pred = np.argmax(probabilities, axis=1)    # Predicted class
        else:
            # For models without predict_proba (e.g., SVM without probability=True)
            y_pred = model.predict(X_unlabeled)
            confidences = np.ones(len(y_pred))  # Assign confidence of 1.0 (no filtering possible)

        # Decode the predicted labels using the LabelEncoder
        try:
            y_pred_labels = label_encoder.inverse_transform(y_pred)
        except Exception as e:
            st.error(f"Error decoding labels: {e}")
            return

        # Create a DataFrame with predictions
        results_df = pd.DataFrame({
            'id_doc': df_unlabeled['id_doc'].values,
            'combined_text': df_unlabeled['combined_text'].values,
            'label': y_pred_labels,
            'confidence': confidences
        })

        # Filter based on minimum confidence
        results_df = results_df[results_df['confidence'] >= min_confidence]

        if results_df.empty:
            st.warning("No predictions meet the minimum confidence threshold.")
            return

        st.write(f"Generated predictions for {len(results_df)} rows with confidence >= {min_confidence}.")

        # Save predictions to CSV
        parquet_base_name = os.path.splitext(selected_file)[0]
        model_name = os.path.splitext(selected_model_file)[0].split('_')[-1]
        output_filename = f"{parquet_base_name}_{model_name}_predictions.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        results_df.to_csv(output_path, index=False)
        st.write(f"Predictions saved to: {output_path}")

        # Display a sample of the results
        st.write("### Sample of Predictions")
        st.dataframe(results_df.head())

if __name__ == "__main__":
    app()