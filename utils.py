import re
import string
import time
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from opensearchpy import OpenSearch, RequestsHttpConnection
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


# Download stopwords if not already downloaded
import nltk
nltk.download('stopwords', quiet=True)

# Text processing functions
def clean_text(text):
    """Cleans the text by removing punctuation, numbers, and extra whitespace."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

def preprocess_texts(texts):
    """Preprocess a list of texts by cleaning and removing stopwords."""
    stop_words = set(stopwords.words('portuguese'))
    preprocessed_texts = []
    for text in texts:
        clean = clean_text(text)
        words = [word for word in clean.split() if word not in stop_words]
        preprocessed_texts.append(" ".join(words))
    return preprocessed_texts

def extract_keywords(texts, top_n=5):
    """Extract top N keywords from the list of texts using TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    sums = tfidf_matrix.sum(axis=0)
    word_scores = [(word, sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
    
    return [word for word, score in sorted_words[:top_n]]

def propose_cluster_names(df):
    """Propose cluster names based on keywords extracted from cluster texts."""
    cluster_names = {}
    for cluster_id in df['cluster'].unique():
        cluster_texts = df[df['cluster'] == cluster_id]['ds_documento_ocr'].tolist()
        preprocessed_texts = preprocess_texts(cluster_texts)
        keywords = extract_keywords(preprocessed_texts)
        cluster_name = " ".join(keywords)
        cluster_names[cluster_id] = cluster_name
        
    return cluster_names

# OpenSearch functions
def connect_opensearch(host, port, timeout=60, max_retries=3):
    """Connect to the OpenSearch instance."""
    es = OpenSearch(
        hosts=[{"host": host, "port": port}],
        scheme="http",
        verify_certs=False,
        connection_class=RequestsHttpConnection,
        timeout=timeout,
        max_retries=max_retries,
        retry_on_timeout=True
    )
    return es

def fetch_documents_old(es, index):
    """Fetch all documents from the OpenSearch index."""
    scroll_timeout = "2m"
    size = 1000
    all_documents = []
    query = {
        "query": {
            "match_all": {}
        },
        "_source": [
            "ds_documento_ocr", "id_documento", "tipo_processo", "label", 
            "embedding_completo"
        ]
    }

    response = es.search(index=index, body=query, scroll=scroll_timeout, size=size)
    scroll_id = response['_scroll_id']
    documents = response['hits']['hits']
    all_documents.extend(documents)

    while len(documents) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
        documents = response['hits']['hits']
        all_documents.extend(documents)

    return all_documents

def fetch_documents(es, index):
    """Fetch all documents from the OpenSearch index."""
    scroll_timeout = "2m"
    size = 1000
    all_documents = []
    query = {
        "query": {
            "match_all": {}
        },
        "_source": [
            "id_doc", "chunk_text", "chunk_embedding", "label"
        ]
    }

    response = es.search(index=index, body=query, scroll=scroll_timeout, size=size)
    scroll_id = response['_scroll_id']
    documents = response['hits']['hits']
    all_documents.extend(documents)

    while len(documents) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
        documents = response['hits']['hits']
        all_documents.extend(documents)

    return all_documents


def create_dataframe(documents):
    """Convert the list of documents into a pandas DataFrame."""
    data = []
    for doc in documents:
        chunk_text = doc["_source"]["chunk_text"].replace("$", "")
        data.append({
            "id_doc": doc["_source"]["id_doc"],
            "chunk_text": chunk_text,
            "chunk_embedding": doc["_source"]["chunk_embedding"],
            "label": doc["_source"].get("label", ""),
        })
    return pd.DataFrame(data)
def create_dataframe2(documents):
    """Convert the list of documents into a pandas DataFrame."""
    data = []
    for doc in documents:
        documento_ocr = doc["_source"]["ds_documento_ocr"].replace("$", "")
        data.append({
            "id_documento": doc["_source"]["id_documento"],
            "ds_documento_ocr": documento_ocr,
            "embedding_completo": doc["_source"]["embedding_completo"],
            "label": doc["_source"].get("label", ""),
        })
    return pd.DataFrame(data)

def persist_labels(es, index, df):
    """Persist the updated labels to the OpenSearch index, ensuring id_documento is unique."""
    updated = False
    for _, row in df.iterrows():
        if row['label']:
            doc_id = row['id_documento']
            label = row['label']

            # Prepare the updated document with the new label
            updated_doc = {
                "doc": {"label": label}
            }
            
            try:
                # Update the document using id_documento as _id
                es.update(index=index, id=doc_id, body=updated_doc)
                updated = True
            except Exception as e:
                logger.error(f"Failed to update document {doc_id}: {e}")
                logger.error(f"Request body: {updated_doc}")
    
    return updated


def calculate_suggested_label(df, n_similares,threshold, perc_filtro):
    df_rotulados = df[df['label'] != ""]
    df_rotulados = df_rotulados.sample(frac=1, random_state=42)
    df_nao_rotulados = df[df['label'] == ""]
    df_nao_rotulados = df_nao_rotulados.sample(frac=perc_filtro/100, random_state=42)
    embeddings_rotulados = np.array(df_rotulados["embedding"].tolist())
    embeddings_nao_rotulados = np.array(df_nao_rotulados["embedding"].tolist())

    rotulos_sugeridos = []
    medias_top_similaridades = []
    contagem_most_common = []
    for embedding_nao_rotulado in embeddings_nao_rotulados:
        similarities = cosine_similarity(embeddings_rotulados, [embedding_nao_rotulado]).flatten()
        similar_indices = np.argsort(similarities)[-n_similares:]
        similar_labels = df_rotulados.iloc[similar_indices]['label']
        most_common_label = Counter(similar_labels).most_common(1)[0][0]
        rotulos_sugeridos.append(most_common_label)
        most_common_count = Counter(similar_labels)[most_common_label]
        contagem_most_common.append(most_common_count)
    
    
    # Média das top n similaridades
        top_similarities = similarities[similar_indices]
        media_top_similaridades = np.mean(top_similarities)
        medias_top_similaridades.append(media_top_similaridades)
        
    df_nao_rotulados['media'] = medias_top_similaridades
    df_nao_rotulados['contagem'] = contagem_most_common
    df_nao_rotulados['rotulo_sugerido'] = rotulos_sugeridos
    df_nao_rotulados = df_nao_rotulados[df_nao_rotulados['contagem'] == n_similares]    
    df_nao_rotulados = df_nao_rotulados[df_nao_rotulados['media'] > threshold].sort_values(by='media', ascending=False)
    df.update(df_nao_rotulados)
    return df_nao_rotulados


def search_similar_embeddings(client, index_name, embeddings, top_k=50):
    responses = []

    for i, embedding in enumerate(embeddings):
        # Criar a consulta KNN para o embedding atual
        query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding_completo": {
                        "vector": embedding,
                        "k": top_k
                    }
                }
            }
        }
        
        # Imprimir o embedding que está sendo processado
        #print(f"Processando embedding {i+1}/{len(embeddings)}: {embedding}")
        
        # Enviar a consulta ao OpenSearch
        response = client.search(index=index_name, body=query)
        
        # Capturar os hits (resultados) retornados
        hits = response['hits']['hits']
        
        # Imprimir o número de resultados retornados
        print(f"Número de resultados retornados: {len(hits)}")
        
        # Adicionar os resultados à lista de respostas
        responses.append(hits)
    
    return responses


def calculate_suggested_label_opt(df, index_name, client, top_k=50):
    # Separar o dataframe entre rotulados e não rotulados
    df_rotulados = df[df['label'] != ""]
    df_nao_rotulados = df[df['label'] == ""]
    
    # Pré-selecionar até três amostras de embeddings para cada rótulo
    label_embeddings = {}
    for label in df_rotulados['label'].unique():
        embeddings = df_rotulados[df_rotulados['label'] == label]['embedding_completo'].head(3).tolist()
        label_embeddings[label] = embeddings
    
    # Dicionário para armazenar resultados de busca
    aggregated_results = defaultdict(list)

    # Realizar a busca de todos os embeddings de cada rótulo em lote
    for label, embeddings_rotulo in label_embeddings.items():
        print(label)
        # Realiza as buscas no OpenSearch para todos os embeddings de um rótulo de uma vez
        responses = search_similar_embeddings(client, index_name, embeddings_rotulo, top_k)
        #print('Responses:',responses)
        # Agrupar os resultados por rótulo
        for response in responses:
            for result in response:
                #print(result)
                #if result['_source']['id_documento'] not in aggregated_results[label]:
                aggregated_results[label].append(result['_source']['id_documento'])  # Usando id_documento

    # Agora vamos atualizar o dataframe com os rótulos sugeridos
    novo_dataframe = pd.DataFrame()

    # Copiar os valores correspondentes de `df_nao_rotulados` para `novo_dataframe`
    for label, doc_ids in aggregated_results.items():
        #print('Lista de ids:',doc_ids)
        # Filtrar as instâncias no dataframe `df_nao_rotulados` que correspondem a `doc_ids`
        filtered_df = df_nao_rotulados[df_nao_rotulados['id_documento'].isin(doc_ids)].copy()
        
        # Atribuir o rótulo sugerido (key do dicionário) para a nova coluna
        filtered_df['rotulo_sugerido'] = label
        
        # Adicionar as linhas filtradas ao novo dataframe
        novo_dataframe = pd.concat([novo_dataframe, filtered_df], ignore_index=True)
    
    #print(novo_dataframe)
    return novo_dataframe

def calculate_suggested_label_neural(df, n_similares, threshold, es_client, index_name):
    df_rotulados = df[df['label'] != ""]
    df_nao_rotulados = df[df['label'] == ""]
    
    # Ensure labeled data is already indexed in OpenSearch
    # You should have a separate process to index `df_rotulados` before calling this function
    
    rotulos_sugeridos = []
    medias_top_similaridades = []
    contagem_most_common = []

    for _, row in df_nao_rotulados.iterrows():
        embedding_nao_rotulado = row['embedding_completo']

        # Query OpenSearch for nearest neighbors
        query = {
            "size": n_similares,
            "query": {
                "knn": {
                    "embedding_completo": {
                        "vector": embedding_nao_rotulado,
                        "k": n_similares
                    }
                }
            }
        }

        response = es_client.search(index=index_name, body=query)

        # Process the OpenSearch response
        similar_labels = [hit['_source']['label'] for hit in response['hits']['hits']]
        similarities = [hit['_score'] for hit in response['hits']['hits']]

        most_common_label = Counter(similar_labels).most_common(1)[0][0]
        rotulos_sugeridos.append(most_common_label)
        most_common_count = Counter(similar_labels)[most_common_label]
        contagem_most_common.append(most_common_count)

        # Calculate mean of top similarities
        media_top_similaridades = np.mean(similarities)
        medias_top_similaridades.append(media_top_similaridades)

    df_nao_rotulados['media'] = medias_top_similaridades
    df_nao_rotulados['contagem'] = contagem_most_common
    df_nao_rotulados['rotulo_sugerido'] = rotulos_sugeridos
    df_nao_rotulados = df_nao_rotulados[df_nao_rotulados['media'] > threshold].sort_values(by='media', ascending=False)
    
    # Update original DataFrame with the suggested labels
    df.update(df_nao_rotulados)
    
    return df_nao_rotulados


def fetch_documents_by_query(es, index, search_string):
    scroll_timeout = "2m"
    size = 1000
    all_documents = []
    
    query = {
        "query": {
            "match": {
                "ds_documento_ocr": search_string
            }
        },
        "_source": [
            "ds_documento_ocr", "id_documento",  "label","embedding_completo","tipo_processo"
        ]
    }

    # Inicia a consulta com scroll
    response = es.search(index=index, body=query, scroll=scroll_timeout, size=size)
    scroll_id = response['_scroll_id']
    documents = response['hits']['hits']

    # Adiciona os documentos recuperados à lista
    all_documents.extend(documents)

    # Continua recuperando documentos até que não restem mais
    while len(documents) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
        documents = response['hits']['hits']
        all_documents.extend(documents)

    return all_documents
    
def fetch_documents_by_tipo(es, index, search_string):
    scroll_timeout = "2m"
    size = 1000
    all_documents = []
    
    query = {
        "query": {
            "match": {
                "tipo": search_string
            }
        },
        "_source": [
            "ds_documento_ocr", "id_documento", "label","embedding_completo","tipo_processo"
        ]
    }

    # Inicia a consulta com scroll
    response = es.search(index=index, body=query, scroll=scroll_timeout, size=size)
    scroll_id = response['_scroll_id']
    documents = response['hits']['hits']

    # Adiciona os documentos recuperados à lista
    all_documents.extend(documents)

    # Continua recuperando documentos até que não restem mais
    while len(documents) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
        documents = response['hits']['hits']
        all_documents.extend(documents)

    return all_documents

def calculate_similarity(df, n_clusters):
    embeddings = np.array(df["embedding"].tolist())
    
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(embeddings)
    df['cluster'] = labels
    
    return df
