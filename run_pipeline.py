
from src.classification.classifier_training import app as classifier_training
from src.classification.classifier import app as classifier
from src.search.suggestion import app as suggestion
from src.search.search import app as search
from src.clustering.clusterer import app as cluster
from src.data.data_source import app as datasource
from src.utilities.converter import app as converter
from src.classification.text_classifier_app import app as text_classifier_app
import streamlit as st
import sys
import os

# Ensure the root directory is in the Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure Streamlit page settings
st.set_page_config(
    page_title="Text Classifier and Clustering", layout="wide")

st.sidebar.title("App Selection")

app_selection = st.sidebar.radio("Select an App", [
    "Home",
    "Datasource",
    "Cluster",
    "Advanced Search",
    "Suggestion",
    "Classifier - Training",
    "Classifier",
    "Convert Parquet to CSV",
    "Text Classifier"
])

if app_selection == "Home":
    st.write("Welcome to the Document Classifier and Clustering App!")
    st.write("Please select an application from the sidebar.")
elif app_selection == "Datasource":
    datasource()
elif app_selection == "Cluster":
    cluster()
elif app_selection == "Advanced Search":
    search()
elif app_selection == "Suggestion":
    suggestion()
elif app_selection == "Classifier - Training":
    classifier_training()
elif app_selection == "Classifier":
    classifier()

elif app_selection == "Convert Parquet to CSV":
    converter()
elif app_selection == "Text Classifier":
    text_classifier_app()