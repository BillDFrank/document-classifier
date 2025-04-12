import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.data.data_source import app as datasource
from src.clustering.clusterer import app as cluster
from src.clustering.cluster_splitter import app as splitcluster
from src.clustering.cluster_mover import app as movecluster
from src.search.search import app as pesquisa
from src.search.suggestion import app as sugestao
from src.search.auto_suggestion import app as sugestao_auto
from src.analysis.outliers import app as outliers
from src.classification.classifier import app as classificador
from src.visualization.plotter import app as plot  # Fixed incorrect import path

st.set_page_config(page_title="Plataforma de Classificação", layout="wide")

# Barra lateral para selecionar plataforma de classificação
st.sidebar.title("Plataforma de Classificação")

# Título principal
st.title("Plataforma de Classificação")

app_selection = st.sidebar.radio("Selecione o App", ["Datasource", "Cluster", "Pesquisa Avançada", "Sugestão", "Sugestão Automática", "Dividir Cluster", "Tratar Outliers", "Move Clusters", "Classificador", "Plot"])

# Renderizar o aplicativo selecionado na área principal
if app_selection == "Datasource":
    datasource()
elif app_selection == "Cluster":
    cluster()
elif app_selection == "Pesquisa Avançada":
    pesquisa()
elif app_selection == "Sugestão":
    sugestao()
elif app_selection == "Sugestão Automática":
    sugestao_auto()
elif app_selection == "Dividir Cluster":
    splitcluster()
elif app_selection == "Tratar Outliers":
    outliers()
elif app_selection == "Move Clusters":
    movecluster()
elif app_selection == "Classificador":
    classificador()
elif app_selection == "Plot":
    plot()