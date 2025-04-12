import streamlit as st
#from streamlit_navigation_bar import st_navbar
from datasource import app as datasource
from cluster import app as cluster
from pesquisa import app as pesquisa
from sugestao import app as sugestao
from sugestao_auto import app as sugestao_auto
from splitcluster import app as splitcluster
from outliers import app as outliers
from movecluster import app as movecluster
from classificador import app as classificador
from plot import app as plot
#st.set_page_config(page_title="Plataforma de Classificação", layout="wide")

# Barra lateral para seleção de aplicativos
#st.sidebar.title("Plataforma de Classificação")

# Título principal
st.title("Plataforma de Classificação")

app_selection = st.sidebar.radio("Selecione o App", ["Datasource","Cluster", "Pesquisa Avançada","Sugestão","Sugestão Automática","Dividir Cluster","Tratar Outliers","Move Clusters", "Classificador","Plot"])


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