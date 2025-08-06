import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories using pathlib
PARQUET_DIR = Path("data") / "processed"
MODELS_DIR = Path("models")

def load_model():
    """Load the pre-trained classifier from the models directory."""
    try:
        model_path = MODELS_DIR / st.session_state.selected_model
        logger.info(f"Loading model from: {model_path}")
        
        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully with joblib")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model with joblib: {e}")
            st.error(f"Failed to load model from {model_path} with joblib: {e}")
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("Model loaded successfully with pickle fallback")
                st.warning("Loaded with pickle as a fallback. Ensure compatibility.")
                return model
            except Exception as e2:
                logger.error(f"Failed to load model with pickle fallback: {e2}")
                st.error(f"Failed to load model with pickle fallback: {e2}")
                return None
                
    except Exception as e:
        logger.error(f"Error in load_model: {e}")
        st.error(f"Error loading model: {e}")
        return None

def suggest_labels(df, model, batch_size):
    """Suggest labels for unlabeled documents in batches, grouped by predicted label and confidence."""
    try:
        logger.info("Starting label suggestion process")
        
        # Filter unlabeled documents
        unlabeled_df = df[df['label'].str.strip() == ""].copy()
        logger.info(f"Found {len(unlabeled_df)} unlabeled documents")
        
        if unlabeled_df.empty or model is None:
            logger.info("No unlabeled documents found or model is None")
            return None

        if 'embedding' not in df.columns:
            logger.error("The 'embedding' column is missing")
            st.error("The 'embedding' column is missing. Ensure embeddings are precomputed.")
            return None

        # Extract embeddings for prediction (only for unlabeled rows)
        try:
            X_unlabeled = np.array(unlabeled_df['embedding'].tolist())
            logger.info(f"Extracted embeddings for {len(X_unlabeled)} documents")
            
            predictions = model.predict(X_unlabeled)
            probabilities = model.predict_proba(X_unlabeled)
            max_probabilities = np.max(probabilities, axis=1)  # Confidence as max probability
            
            logger.info("Model predictions completed successfully")
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            st.error(f"Error during model prediction: {e}")
            return None

        # Map numeric predictions to string labels using the label mapping
        try:
            label_mapping = st.session_state.label_mapping
            predicted_labels = [label_mapping[p] for p in predictions]
            unlabeled_df['suggested_label'] = predicted_labels
            unlabeled_df['confidence'] = max_probabilities
            logger.info("Labels mapped successfully")
        except Exception as e:
            logger.error(f"Error mapping labels: {e}")
            st.error(f"Error mapping labels: {e}")
            return None

        # Group by predicted label and sort by confidence within each group
        try:
            grouped_df = unlabeled_df.groupby('suggested_label', group_keys=False).apply(
                lambda x: x.sort_values('confidence', ascending=False)
            )
            logger.info("Documents grouped and sorted by confidence")
        except Exception as e:
            logger.error(f"Error grouping documents: {e}")
            st.error(f"Error grouping documents: {e}")
            return None

        # Split into batches based on grouped labels
        try:
            batches = []
            for label, group in grouped_df.groupby('suggested_label'):
                label_batches = [group[i:i + batch_size] for i in range(0, len(group), batch_size)]
                batches.extend(label_batches)
            
            logger.info(f"Created {len(batches)} batches")
            return batches
            
        except Exception as e:
            logger.error(f"Error creating batches: {e}")
            st.error(f"Error creating batches: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error in suggest_labels: {e}")
        st.error(f"Error suggesting labels: {e}")
        return None

def app():
    st.title("🏷️ Category Suggestion")
    st.write("Predicts and suggests labels for unlabeled documents.")

    st.sidebar.title("⚙️ Settings")

    # List all .parquet files in the data/processed directory
    try:
        if not PARQUET_DIR.exists():
            logger.error(f"Directory '{PARQUET_DIR}' not found")
            st.error(f"Directory '{PARQUET_DIR}' not found. Generate embeddings first.")
            return

        parquet_files = [f.name for f in PARQUET_DIR.glob("*.parquet")]
        if not parquet_files:
            logger.error(f"No Parquet files found in '{PARQUET_DIR}'")
            st.error(f"No Parquet files found in '{PARQUET_DIR}'. Generate embeddings first.")
            return

        # Dropdown for selecting Parquet file
        selected_parquet = st.sidebar.selectbox(
            "Select a Parquet file",
            parquet_files,
            help="Choose a Parquet file generated by the Datasource app."
        )
        parquet_file = PARQUET_DIR / selected_parquet
        logger.info(f"Selected parquet file: {parquet_file}")
        
    except Exception as e:
        logger.error(f"Error listing parquet files: {e}")
        st.error(f"Error listing parquet files: {e}")
        return

    # List all .pkl model files in the models directory
    try:
        if not MODELS_DIR.exists():
            logger.error(f"Directory '{MODELS_DIR}' not found")
            st.error(f"Directory '{MODELS_DIR}' not found. Train a model first.")
            return

        model_files = [f.name for f in MODELS_DIR.glob("*.pkl")]
        if not model_files:
            logger.error(f"No .pkl model files found in '{MODELS_DIR}'")
            st.error(f"No .pkl model files found in '{MODELS_DIR}'. Train a model first.")
            return

        # Dropdown for selecting classifier model
        selected_model = st.sidebar.selectbox(
            "Select a Classifier Model",
            model_files,
            help="Choose a trained model (.pkl) from the models directory."
        )
        logger.info(f"Selected model: {selected_model}")
        
    except Exception as e:
        logger.error(f"Error listing model files: {e}")
        st.error(f"Error listing model files: {e}")
        return

    # Sidebar inputs
    batch_size = st.sidebar.slider("Maximum Elements per Batch", 1, 30, 10)
    submit_button = st.sidebar.button("Submit")

    # Initialize session state
    try:
        if 'full_df' not in st.session_state or st.session_state.get('last_parquet_file') != str(parquet_file) or st.session_state.get('last_model') != selected_model:
            logger.info("Loading new parquet file or model")
            
            # Load the Parquet file at the start or when the file/model changes
            full_df = pd.read_parquet(parquet_file)
            logger.info(f"Loaded parquet file with {len(full_df)} rows")
            
            if 'combined_text' not in full_df.columns:
                logger.error("The 'combined_text' column is missing")
                st.error("The 'combined_text' column is missing from the Parquet file.")
                return
                
            full_df['combined_text'] = full_df['combined_text'].astype(str)
            if 'label' not in full_df.columns:
                full_df['label'] = ""
            full_df['label'] = full_df['label'].astype(str).fillna("")
            
            # Do not filter out blank combined_text rows to preserve all rows
            if full_df.empty:
                logger.error("No rows found in the Parquet file")
                st.error("No rows found in the Parquet file.")
                return
                
            st.session_state.full_df = full_df
            st.session_state.last_parquet_file = str(parquet_file)
            st.session_state.selected_model = selected_model
            st.session_state.last_model = selected_model
            st.session_state.current_batch = 0
            st.session_state.batches = None
            st.session_state.label_mapping = None
            
            logger.info("Session state initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing session state: {e}")
        st.error(f"Error loading data: {e}")
        return

    # Initialize session state variables
    try:
        if 'current_batch' not in st.session_state:
            st.session_state.current_batch = 0
        if 'possible_labels' not in st.session_state:
            st.session_state.possible_labels = []
        if 'batches' not in st.session_state:
            st.session_state.batches = None
        if 'label_mapping' not in st.session_state:
            st.session_state.label_mapping = None
    except Exception as e:
        logger.error(f"Error initializing session state variables: {e}")
        st.error(f"Error initializing session state: {e}")
        return

    # Load the model and predict labels when submitting
    if submit_button:
        try:
            logger.info("Submit button clicked, loading model")
            model = load_model()
            if model is None:
                logger.error("Failed to load model")
                return

            # Create a mapping from numeric labels to string labels
            try:
                existing_labels = sorted([label for label in st.session_state.full_df['label'].unique() if label])
                model_classes = list(model.classes_)
                
                if len(model_classes) != len(existing_labels):
                    logger.error(f"Model classes mismatch: {len(model_classes)} vs {len(existing_labels)}")
                    st.error(f"Number of model classes ({len(model_classes)}) does not match number of unique labels ({len(existing_labels)}). Please ensure the model was trained with the correct labels.")
                    return

                # Map numeric model classes to string labels (assumes order matches)
                label_mapping = {num_label: str_label for num_label, str_label in zip(model_classes, existing_labels)}
                st.session_state.label_mapping = label_mapping
                logger.info("Label mapping created successfully")

                # Get possible labels (model classes mapped to strings + existing labels)
                mapped_classes = [label_mapping[num_label] for num_label in model_classes]
                possible_labels = sorted(list(set(mapped_classes + existing_labels)))
                st.session_state.possible_labels = possible_labels
                logger.info(f"Possible labels determined: {len(possible_labels)} labels")

                # Suggest labels and create batches (only for unlabeled rows)
                batches = suggest_labels(st.session_state.full_df, model, batch_size)
                st.session_state.batches = batches
                st.session_state.current_batch = 0
                logger.info("Label suggestions completed")
                
            except Exception as e:
                logger.error(f"Error during model setup: {e}")
                st.error(f"Error during model setup: {e}")
                
        except Exception as e:
            logger.error(f"Error processing submit: {e}")
            st.error(f"Error processing submit: {e}")

    # Display and labeling interface
    try:
        if st.session_state.batches is None or not st.session_state.batches:
            st.info("No unlabeled documents found or all documents are labeled.")
            return

        batches = st.session_state.batches
        total_batches = len(batches)
        current_batch = st.session_state.current_batch

        # Navigation buttons
        st.header("📋 Batch Navigation")
        st.write(f"Batch: {current_batch + 1} of {total_batches}")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("⬅️ Previous Batch"):
                st.session_state.current_batch = max(0, current_batch - 1)
        with col2:
            if st.button("➡️ Next Batch"):
                st.session_state.current_batch = min(total_batches - 1, current_batch + 1)

        # Update current batch
        current_batch = st.session_state.current_batch
        batch_df = batches[current_batch].copy()
        current_label = batch_df['suggested_label'].iloc[0]  # All documents in this batch have the same predicted label

        # Display current batch
        st.header(f"📄 Batch {current_batch + 1} of {total_batches} (Predicted Label: {current_label})")
        st.write(f"Displaying up to {len(batch_df)} items sorted by confidence")

        # Display batch with manual label editing
        possible_labels = st.session_state.possible_labels
        for idx, row in batch_df.iterrows():
            col1, col2, col3 = st.columns([6, 2, 2])
            with col1:
                text_preview = row['combined_text'][:500] if len(str(row['combined_text'])) > 500 else str(row['combined_text'])
                st.write(f"[{row['id_doc']}] - {text_preview}...")
            with col2:
                suggested_label = row['suggested_label']
                new_label = st.selectbox(
                    "Suggested Label",
                    possible_labels,
                    index=possible_labels.index(suggested_label) if suggested_label in possible_labels else 0,
                    key=f"label_{idx}"
                )
            with col3:
                confidence_display = f"{row['confidence']:.2f}" if pd.notnull(row['confidence']) else "N/A"
                st.write(f"Confidence: {confidence_display}")

            # Update the label in the batch DataFrame if changed
            if new_label != suggested_label:
                batch_df.at[idx, 'suggested_label'] = new_label

        if st.button("💾 Save"):
            try:
                logger.info("Saving labels for current batch")
                
                # Update the full DataFrame with the new labels from the current batch
                for idx in batch_df.index:
                    st.session_state.full_df.at[idx, 'label'] = batch_df.at[idx, 'suggested_label']
                    
                # Update the batch in session state
                batches[current_batch] = batch_df
                st.session_state.batches = batches

                # Save the full DataFrame to the Parquet file, preserving all rows
                st.session_state.full_df.to_parquet(parquet_file, index=False)
                st.success("✅ Labels for the current batch updated successfully in the Parquet file!")
                logger.info("Labels saved successfully")
                
            except Exception as e:
                logger.error(f"Error saving labels: {e}")
                st.error(f"Error updating file: {e}")

        # Refresh suggestions after labeling
        if st.button("🔄 Refresh Suggestions"):
            try:
                logger.info("Refreshing suggestions")
                model = load_model()
                if model is not None:
                    batches = suggest_labels(st.session_state.full_df, model, batch_size)
                    st.session_state.batches = batches
                    st.session_state.current_batch = 0
                    st.success("✅ Suggestions refreshed!")
                    logger.info("Suggestions refreshed successfully")
            except Exception as e:
                logger.error(f"Error refreshing suggestions: {e}")
                st.error(f"Error refreshing suggestions: {e}")
                
    except Exception as e:
        logger.error(f"Error in display interface: {e}")
        st.error(f"Error displaying interface: {e}")

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        logger.error(f"Fatal error in app: {e}")
        st.error(f"Fatal error: {e}")