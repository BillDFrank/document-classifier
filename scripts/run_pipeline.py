from src.visualization.plotter import app as plot
from src.classification.classifier import app as classifier
from src.analysis.outliers import app as outliers
from src.search.auto_suggestion import app as auto_suggestion
from src.search.suggestion import app as suggestion
from src.search.search import app as search
from src.clustering.cluster_mover import app as movecluster
from src.clustering.cluster_splitter import app as splitcluster
from src.clustering.clusterer import app as cluster
from src.data.data_source import app as datasource
import streamlit as st
import sys
import os

# Ensure the root directory is in the Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Configure Streamlit page settings
st.set_page_config(page_title="Classification Platform", layout="wide")

st.sidebar.title("Classification Platform")
st.title("Classification Platform")

app_selection = st.sidebar.radio("Select an App", [
    "Datasource",
    "Cluster",
    "Advanced Search",
    "Suggestion",
    "Auto Suggestion",
    "Split Cluster",
    "Handle Outliers",
    "Move Clusters",
    "Classifier",
    "Plot"
])

if app_selection == "Datasource":
    datasource()
elif app_selection == "Cluster":
    cluster()
elif app_selection == "Advanced Search":
    search()
elif app_selection == "Suggestion":
    suggestion()
elif app_selection == "Auto Suggestion":
    auto_suggestion()
elif app_selection == "Split Cluster":
    splitcluster()
elif app_selection == "Handle Outliers":
    outliers()
elif app_selection == "Move Clusters":
    movecluster()
elif app_selection == "Classifier":
    classifier()
elif app_selection == "Plot":
    plot()
