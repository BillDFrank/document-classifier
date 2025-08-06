import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import joblib  # For saving models
import json
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import plotly.figure_factory as ff

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get base directory from environment variable or use default
BASE_DIR = Path(os.getenv('PROJECT_ROOT', '.')).resolve()

# Directory where Parquet files are saved
PARQUET_DIR = BASE_DIR / "data" / "processed"
# Directory to save models
MODEL_DIR = BASE_DIR / "models"

# Create the models directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def app():
    st.title('Process Classification')

    # List available Parquet files
    try:
        parquet_files = [f for f in os.listdir(PARQUET_DIR) if f.endswith('.parquet')]
        if not parquet_files:
            st.error(f"No Parquet files found in {PARQUET_DIR}")
            logger.error(f"No Parquet files found in {PARQUET_DIR}")
            return
    except Exception as e:
        st.error(f"Error accessing {PARQUET_DIR}: {str(e)}")
        logger.error(f"Error accessing {PARQUET_DIR}: {str(e)}")
        return

    selected_file = st.sidebar.selectbox("Select a Parquet file", parquet_files)
    file_path = PARQUET_DIR / selected_file
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} records from {file_path}")
    except Exception as e:
        st.error(f"Error loading parquet file {file_path}: {str(e)}")
        logger.error(f"Error loading parquet file {file_path}: {str(e)}")
        return

    # Filter labeled data and remove rows without embeddings
    df_labeled = df[df['label'].notnull() & (df['label'].str.strip() != '')]
    df_labeled = df_labeled.dropna(subset=['embedding'])
    if df_labeled.empty:
        st.warning("No labeled data with embeddings found for training.")
        return

    # Display dataset information
    st.write("### Dataset Information")
    st.write(f"Total labeled instances: {len(df_labeled)}")
    st.write("Class distribution:")
    st.write(df_labeled['label'].value_counts().to_dict())

    # Dropdown for model selection
    model_options = ["Logistic Regression", "SVM", "Random Forest", "Neural Network", "KNN"]
    selected_model = st.sidebar.selectbox("Select a model to train", model_options)

    # Default hyperparameters (will be updated based on user input)
    threshold = 0.5  # For Logistic Regression
    logreg_params = {'C': 1.0, 'max_iter': 1000}
    svm_params = {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
    rf_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
    nn_params = {'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001, 'max_iter': 1000}
    knn_params = {'n_neighbors': 5, 'weights': 'uniform', 'p': 2}

    # Conditionally show hyperparameter options based on selected model
    if selected_model == "Logistic Regression":
        st.sidebar.subheader("Logistic Regression Hyperparameters")
        logreg_params['C'] = st.sidebar.slider("C (Regularization Strength)", 0.01, 10.0, 1.0, step=0.01)
        logreg_params['max_iter'] = st.sidebar.number_input("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
        if len(df_labeled['label'].unique()) > 2:
            st.sidebar.warning("Threshold adjustment is only applicable for binary classification. It will be ignored for multi-class tasks.")

    elif selected_model == "SVM":
        st.sidebar.subheader("SVM Hyperparameters")
        svm_params['C'] = st.sidebar.slider("C (Regularization Parameter)", 0.01, 10.0, 1.0, step=0.01)
        svm_params['kernel'] = st.sidebar.selectbox("Kernel", ["linear", "rbf"])
        if svm_params['kernel'] == "rbf":
            svm_params['gamma'] = st.sidebar.selectbox("Gamma (Kernel Coefficient)", ["scale", "auto", 0.001, 0.01, 0.1, 1.0], index=0)

    elif selected_model == "Random Forest":
        st.sidebar.subheader("Random Forest Hyperparameters")
        rf_params['n_estimators'] = st.sidebar.number_input("Number of Trees", min_value=10, max_value=500, value=100, step=10)
        max_depth_options = [None] + list(range(10, 51, 10))
        rf_params['max_depth'] = st.sidebar.selectbox("Max Depth", max_depth_options, index=0, format_func=lambda x: "None" if x is None else x)
        rf_params['min_samples_split'] = st.sidebar.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1)

    elif selected_model == "Neural Network":
        st.sidebar.subheader("Neural Network Hyperparameters")
        nn_params['hidden_layer_sizes'] = (st.sidebar.number_input("Hidden Layer Size", min_value=10, max_value=500, value=100, step=10),)
        nn_params['learning_rate_init'] = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001, step=0.0001)
        nn_params['max_iter'] = st.sidebar.number_input("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)

    elif selected_model == "KNN":
        st.sidebar.subheader("KNN Hyperparameters")
        knn_params['n_neighbors'] = st.sidebar.number_input("Number of Neighbors", min_value=1, max_value=50, value=5, step=1)
        knn_params['weights'] = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        knn_params['p'] = st.sidebar.selectbox("Distance Metric (p)", [1, 2], format_func=lambda x: "Manhattan (p=1)" if x == 1 else "Euclidean (p=2)")

    # Button to start training
    if st.sidebar.button("Train Model"):
        # Prepare features and labels
        try:
            X = np.array(df_labeled['embedding'].tolist())
            y = np.array(df_labeled['label'].tolist())
            
            if len(X) == 0 or len(y) == 0:
                st.error("No valid data found for training")
                logger.error("No valid data found for training")
                return
                
            logger.info(f"Prepared {len(X)} samples for training")
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            logger.error(f"Error preparing data: {str(e)}")
            return
    
        # Encode labels
        try:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            is_binary = len(label_encoder.classes_) == 2
            logger.info(f"Encoded {len(label_encoder.classes_)} classes: {label_encoder.classes_}")
        except Exception as e:
            st.error(f"Error encoding labels: {str(e)}")
            logger.error(f"Error encoding labels: {str(e)}")
            return

        # Set up 5-fold cross-validation
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # Lists to store metrics for each fold
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        confusion_matrices = []
        all_y_test = []
        all_y_pred = []

        # Display progress bar
        st.write("Training the model with 5-fold cross-validation...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Get number of classes and their labels
        n_classes = len(label_encoder.classes_)
        class_labels = list(range(n_classes))  # Use numeric indices for consistent matrix sizes
        st.write(f"Number of classes detected: {n_classes}")
        
        # If more than 2 classes, disable threshold selection for Logistic Regression
        if n_classes > 2 and selected_model == "Logistic Regression":
            threshold = 0.5  # Reset to default
            st.warning("Threshold adjustment is not applicable for multi-class classification. Using default decision boundary.")

        # Estimate training time
        n_samples = len(X)
        estimated_times_per_fold = {
            "Logistic Regression": 0.1 * n_samples / 1000,
            "SVM": 0.5 * n_samples / 1000,
            "Random Forest": 0.3 * n_samples / 1000,
            "Neural Network": 1.0 * n_samples / 1000,
            "KNN": 0.05 * n_samples / 1000
        }
        estimated_time_per_fold = estimated_times_per_fold.get(selected_model, 1.0)
        total_estimated_time = estimated_time_per_fold * k
        status_text.text(f"Estimated time to finish: {total_estimated_time:.2f} seconds")

        # Cross-validation loop
        start_time = time.time()
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]

            # Train the selected model with user-specified hyperparameters
            if selected_model == "Logistic Regression":
                if n_classes > 2:
                    model = LogisticRegression(C=logreg_params['C'], max_iter=logreg_params['max_iter'], 
                                             multi_class='multinomial', solver='lbfgs', random_state=42)
                else:
                    model = LogisticRegression(C=logreg_params['C'], max_iter=logreg_params['max_iter'], random_state=42)
            elif selected_model == "SVM":
                if n_classes > 2:
                    model = SVC(C=svm_params['C'], kernel=svm_params['kernel'], gamma=svm_params['gamma'],
                              decision_function_shape='ovr', random_state=42)
                else:
                    model = SVC(C=svm_params['C'], kernel=svm_params['kernel'], gamma=svm_params['gamma'], random_state=42)
            elif selected_model == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=rf_params['n_estimators'],
                    max_depth=rf_params['max_depth'],
                    min_samples_split=rf_params['min_samples_split'],
                    random_state=42
                )
            elif selected_model == "Neural Network":
                # Adjust hidden layer sizes based on number of classes
                hidden_size = max(nn_params['hidden_layer_sizes'][0], n_classes * 10)
                model = MLPClassifier(
                    hidden_layer_sizes=(hidden_size,),
                    learning_rate_init=nn_params['learning_rate_init'],
                    max_iter=nn_params['max_iter'],
                    random_state=42
                )
            elif selected_model == "KNN":
                model = KNeighborsClassifier(
                    n_neighbors=min(knn_params['n_neighbors'], len(X_train)),
                    weights=knn_params['weights'],
                    p=knn_params['p']
                )

            model.fit(X_train, y_train)

            # Get predictions
            if n_classes == 2 and selected_model == "Logistic Regression" and threshold != 0.5:
                # Only use threshold for binary Logistic Regression when custom threshold is set
                y_scores = model.predict_proba(X_test)[:, 1]
                y_pred = (y_scores >= threshold).astype(int)
            else:
                # For multi-class or other models, use standard predict
                y_pred = model.predict(X_test)

            # Evaluate on the test fold
            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, average='weighted'))
            recalls.append(recall_score(y_test, y_pred, average='weighted'))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            
            # Create confusion matrix with fixed size for all classes
            conf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
            confusion_matrices.append(conf_matrix)

            # Collect predictions for aggregated classification report
            all_y_test.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

            # Update progress bar
            progress_bar.progress((fold + 1) / k)

        # Update progress bar to 100% and clear estimated time
        progress_bar.progress(1.0)
        status_text.text("Training completed!")

        # Save the model (using the last trained model from the cross-validation loop)
        try:
            parquet_base_name = os.path.splitext(selected_file)[0]  # Remove .parquet extension
            model_filename = f"{parquet_base_name}_{selected_model.replace(' ', '')}.joblib"
            model_path = MODEL_DIR / model_filename
            joblib.dump(model, model_path)
            logger.info(f"Model saved successfully: {model_path}")

            # Save the LabelEncoder
            label_encoder_filename = f"{parquet_base_name}_{selected_model.replace(' ', '')}_label_encoder.joblib"
            label_encoder_path = MODEL_DIR / label_encoder_filename
            joblib.dump(label_encoder, label_encoder_path)
            logger.info(f"LabelEncoder saved successfully: {label_encoder_path}")
        except Exception as e:
            st.error(f"Error saving model or label encoder: {str(e)}")
            logger.error(f"Error saving model or label encoder: {str(e)}")
            return

        st.write(f"Model saved as: {model_path}")
        st.write(f"LabelEncoder saved as: {label_encoder_path}")

        # Compute average metrics
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        avg_recall = np.mean(recalls)
        std_recall = np.std(recalls)
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        # Convert confusion matrices to numpy array and sum
        confusion_matrices = np.array(confusion_matrices)
        agg_conf_matrix = np.sum(confusion_matrices, axis=0)

        # Get original labels for confusion matrix and classification report
        class_names = label_encoder.classes_.tolist()

        # Decode all predictions and true labels for the aggregated classification report
        all_y_test_labels = label_encoder.inverse_transform(all_y_test)
        all_y_pred_labels = label_encoder.inverse_transform(all_y_pred)

        # Display metrics
        st.write(f"### Model Evaluation: {selected_model} (5-Fold Cross-Validation)")
        st.write(f"**Average Accuracy**: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
        st.write(f"**Average Precision**: {avg_precision:.4f} (±{std_precision:.4f})")
        st.write(f"**Average Recall**: {avg_recall:.4f} (±{std_recall:.4f})")
        st.write(f"**Average F1-Score**: {avg_f1:.4f} (±{std_f1:.4f})")
    
        try:
            # Create and display confusion matrix using Plotly
            st.write("**Aggregate Confusion Matrix**:")
            fig = ff.create_annotated_heatmap(
                z=agg_conf_matrix,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                showscale=True,
                annotation_text=agg_conf_matrix.astype(str)
            )
            fig.update_layout(
                title="Aggregate Confusion Matrix",
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                width=500,
                height=500
            )
            st.plotly_chart(fig)
    
            # Display aggregated classification report
            st.write("**Aggregated Classification Report (Across All Folds)**:")
            report = classification_report(all_y_test_labels, all_y_pred_labels, target_names=class_names)
            st.text(report)
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")
            logger.error(f"Error displaying results: {str(e)}")

if __name__ == "__main__":
    app()