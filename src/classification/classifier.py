import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory paths using pathlib.Path for cross-platform compatibility
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PARQUET_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def app():
    st.title('Classifier: Predict Labels for Unlabeled Data')

    # List available Parquet files
    try:
        parquet_files = [f.name for f in PARQUET_DIR.glob('*.parquet')]
        if not parquet_files:
            st.error(f"No Parquet files found in {PARQUET_DIR}")
            return
    except Exception as e:
        st.error(f"Error accessing parquet directory: {e}")
        return

    # List available models - support both .pkl and .joblib extensions
    try:
        model_files = [
            f.name for f in MODEL_DIR.glob('*.pkl')
            if not f.name.endswith('_label_encoder.pkl')
        ] + [
            f.name for f in MODEL_DIR.glob('*.joblib')
            if not f.name.endswith('_label_encoder.joblib')
        ]
        if not model_files:
            st.error(f"No models found in {MODEL_DIR}. Train a model first using Classifier Training.")
            return
    except Exception as e:
        st.error(f"Error accessing models directory: {e}")
        return

    # Sidebar inputs
    selected_file = st.sidebar.selectbox("Select a Parquet file", parquet_files)
    selected_model_file = st.sidebar.selectbox("Select a Model", model_files)
    min_confidence = st.sidebar.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)
    
    # Add option to choose prediction scope
    prediction_scope = st.sidebar.radio(
        "Prediction Scope",
        ["Only Unlabeled Rows", "All Rows"],
        help="Choose whether to predict labels for all rows or only unlabeled rows"
    )

    # Button to start prediction
    if st.sidebar.button("Predict Labels"):
        # Load the Parquet file
        file_path = PARQUET_DIR / selected_file
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            st.error(f"Error loading parquet file: {e}")
            return

        # Filter rows based on prediction scope
        if prediction_scope == "Only Unlabeled Rows":
            df_to_predict = df[
                (df['combined_text'].notnull()) & 
                (df['combined_text'].str.strip() != '') & 
                (df['label'].isnull() | (df['label'].str.strip() == ''))
            ]
        else:  # "All Rows"
            df_to_predict = df[
                (df['combined_text'].notnull()) & 
                (df['combined_text'].str.strip() != '')
            ]

        # Check for embedding column and drop rows with missing embeddings
        if 'embedding' not in df_to_predict.columns:
            st.error("The Parquet file does not contain an 'embedding' column.")
            return
        df_to_predict = df_to_predict.dropna(subset=['embedding'])

        if df_to_predict.empty:
            st.warning(f"No {'unlabeled ' if prediction_scope == 'Only Unlabeled Rows' else ''}data with embeddings found for prediction.")
            return

        st.write(f"Found {len(df_to_predict)} rows with embeddings for prediction.")

        # Load the model
        model_path = MODEL_DIR / selected_model_file
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            logger.error(f"Failed to load model from {model_path}: {e}")
            return

        # Load the corresponding LabelEncoder
        base_name = os.path.splitext(selected_model_file)[0]
        label_encoder_filename = f"{base_name}_label_encoder.pkl"
        label_encoder_path = MODEL_DIR / label_encoder_filename
        
        # Also check for .joblib extension for label encoder
        if not label_encoder_path.exists():
            label_encoder_filename = f"{base_name}_label_encoder.joblib"
            label_encoder_path = MODEL_DIR / label_encoder_filename
            
        try:
            label_encoder = joblib.load(label_encoder_path)
        except Exception as e:
            st.error(f"Error loading LabelEncoder: {e}")
            logger.error(f"Failed to load label encoder from {label_encoder_path}: {e}")
            return

        # Prepare features
        X_to_predict = np.array(df_to_predict['embedding'].tolist())

        # Get predictions and confidences
        if hasattr(model, "predict_proba"):
            # Get probabilities
            probabilities = model.predict_proba(X_to_predict)
            confidences = np.max(probabilities, axis=1)  # Maximum probability as confidence
            y_pred = np.argmax(probabilities, axis=1)    # Predicted class
        else:
            # For models without predict_proba (e.g., SVM without probability=True)
            y_pred = model.predict(X_to_predict)
            confidences = np.ones(len(y_pred))  # Assign confidence of 1.0 (no filtering possible)

        # Decode the predicted labels using the LabelEncoder
        try:
            y_pred_labels = label_encoder.inverse_transform(y_pred)
        except Exception as e:
            st.error(f"Error decoding labels: {e}")
            return

        # Create a DataFrame with predictions
        results_df = pd.DataFrame({
            'id_doc': df_to_predict['id_doc'].values,
            'combined_text': df_to_predict['combined_text'].values,
            'original_label': df_to_predict['label'].values if 'label' in df_to_predict.columns else None,
            'predicted_label': y_pred_labels,
            'confidence': confidences
        })

        # Filter based on minimum confidence
        results_df = results_df[results_df['confidence'] >= min_confidence]

        if results_df.empty:
            st.warning("No predictions meet the minimum confidence threshold.")
            return

        st.write(f"Generated predictions for {len(results_df)} rows with confidence >= {min_confidence}.")

        # Display the number of predictions for each class
        st.write("### Prediction Distribution by Class")
        class_counts = results_df['predicted_label'].value_counts().to_dict()
        st.write(class_counts)

        # If predicting all rows, show comparison with original labels
        if prediction_scope == "All Rows" and 'original_label' in results_df.columns:
            st.write("### Comparison with Original Labels")
            # Calculate accuracy for rows that had original labels
            mask = results_df['original_label'].notna() & (results_df['original_label'].str.strip() != '')
            if mask.any():
                accuracy = (results_df.loc[mask, 'original_label'] == results_df.loc[mask, 'predicted_label']).mean()
                st.write(f"Accuracy on previously labeled data: {accuracy:.2%}")
                
                # Show confusion matrix for rows with original labels
                # Get unique labels and ensure they're in the same order as the label encoder
                unique_labels = label_encoder.classes_
                
                # Create confusion matrix
                conf_matrix = confusion_matrix(
                    results_df.loc[mask, 'original_label'],
                    results_df.loc[mask, 'predicted_label'],
                    labels=unique_labels
                )
                
                # Convert labels to strings to ensure they're compatible with plotly
                label_names = [str(label) for label in unique_labels]
                
                # Create the heatmap
                fig = ff.create_annotated_heatmap(
                    z=conf_matrix,
                    x=label_names,
                    y=label_names,
                    colorscale='Blues',
                    showscale=True,
                    annotation_text=conf_matrix.astype(str)
                )
                
                # Update layout
                fig.update_layout(
                    title="Confusion Matrix (Original vs Predicted Labels)",
                    xaxis_title="Predicted Label",
                    yaxis_title="Original Label",
                    width=500,
                    height=500
                )
                
                # Update x and y axis to show all labels
                fig.update_xaxes(tickangle=45)
                fig.update_yaxes(tickangle=0)
                
                st.plotly_chart(fig)

        # Save predictions to CSV
        parquet_base_name = os.path.splitext(selected_file)[0]
        model_name = os.path.splitext(selected_model_file)[0].split('_')[-1]
        output_filename = f"{parquet_base_name}_{model_name}_predictions.csv"
        output_path = OUTPUT_DIR / output_filename
        
        try:
            results_df.to_csv(output_path, index=False)
            st.success(f"Predictions saved to: {output_path}")
            logger.info(f"Successfully saved {len(results_df)} predictions to {output_path}")
        except Exception as e:
            st.error(f"Error saving predictions: {e}")
            logger.error(f"Failed to save predictions to {output_path}: {e}")
            return

        # Display a sample of the results
        st.write("### Sample of Predictions")
        st.dataframe(results_df.head())

if __name__ == "__main__":
    app()