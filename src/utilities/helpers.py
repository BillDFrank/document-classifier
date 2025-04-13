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
from collections import Counter, defaultdict

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
    stop_words = set(stopwords.words('english'))
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
    word_scores = [(word, sums[0, idx])
                   for word, idx in vectorizer.vocabulary_.items()]
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)

    return [word for word, score in sorted_words[:top_n]]


def propose_cluster_names(df):
    """Propose cluster names based on keywords extracted from cluster texts."""
    cluster_names = {}
    for cluster_id in df['cluster'].unique():
        cluster_texts = df[df['cluster'] ==
                           cluster_id]['document_content'].tolist()
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
            "document_content", "document_id", "process_type", "label",
            "full_embedding"
        ]
    }

    response = es.search(index=index, body=query,
                         scroll=scroll_timeout, size=size)
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
            "doc_id", "chunk_text", "chunk_embedding", "label"
        ]
    }

    response = es.search(index=index, body=query,
                         scroll=scroll_timeout, size=size)
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
            "doc_id": doc["_source"]["doc_id"],
            "chunk_text": chunk_text,
            "chunk_embedding": doc["_source"]["chunk_embedding"],
            "label": doc["_source"].get("label", ""),
        })
    return pd.DataFrame(data)


def create_dataframe2(documents):
    """Convert the list of documents into a pandas DataFrame."""
    data = []
    for doc in documents:
        document_content = doc["_source"]["document_content"].replace("$", "")
        data.append({
            "document_id": doc["_source"]["document_id"],
            "document_content": document_content,
            "full_embedding": doc["_source"]["full_embedding"],
            "label": doc["_source"].get("label", ""),
        })
    return pd.DataFrame(data)


def persist_labels(es, index, df):
    """Persist the updated labels to the OpenSearch index, ensuring document_id is unique."""
    updated = False
    for _, row in df.iterrows():
        if row['label']:
            doc_id = row['document_id']
            label = row['label']

            updated_doc = {
                "doc": {"label": label}
            }

            try:
                es.update(index=index, id=doc_id, body=updated_doc)
                updated = True
            except Exception as e:
                print(f"Failed to update document {doc_id}: {e}")

    return updated


def calculate_suggested_label(df, n_similares, threshold, perc_filter):
    """Calculate suggested labels for unlabeled data based on similarity to labeled data."""
    df_labeled = df[df['label'] != ""]
    df_labeled = df_labeled.sample(frac=1, random_state=42)
    df_unlabeled = df[df['label'] == ""]
    df_unlabeled = df_unlabeled.sample(frac=perc_filter/100, random_state=42)
    embeddings_labeled = np.array(df_labeled["embedding"].tolist())
    embeddings_unlabeled = np.array(df_unlabeled["embedding"].tolist())

    suggested_labels = []
    mean_top_similarities = []
    most_common_counts = []
    for embedding_unlabeled in embeddings_unlabeled:
        similarities = cosine_similarity(
            embeddings_labeled, [embedding_unlabeled]).flatten()
        similar_indices = np.argsort(similarities)[-n_similares:]
        similar_labels = df_labeled.iloc[similar_indices]['label']
        most_common_label = Counter(similar_labels).most_common(1)[0][0]
        suggested_labels.append(most_common_label)
        most_common_count = Counter(similar_labels)[most_common_label]
        most_common_counts.append(most_common_count)

        top_similarities = similarities[similar_indices]
        mean_similarity = np.mean(top_similarities)
        mean_top_similarities.append(mean_similarity)

    df_unlabeled['mean_similarity'] = mean_top_similarities
    df_unlabeled['count'] = most_common_counts
    df_unlabeled['suggested_label'] = suggested_labels
    df_unlabeled = df_unlabeled[df_unlabeled['count'] == n_similares]
    df_unlabeled = df_unlabeled[df_unlabeled['mean_similarity'] > threshold].sort_values(
        by='mean_similarity', ascending=False)
    df.update(df_unlabeled)
    return df_unlabeled


def search_similar_embeddings(client, index_name, embeddings, top_k=50):
    """Search for similar embeddings in OpenSearch using KNN."""
    responses = []

    for i, embedding in enumerate(embeddings):
        query = {
            "size": top_k,
            "query": {
                "knn": {
                    "full_embedding": {
                        "vector": embedding,
                        "k": top_k
                    }
                }
            }
        }

        response = client.search(index=index_name, body=query)
        hits = response['hits']['hits']
        responses.append(hits)

    return responses


def calculate_suggested_label_opt(df, index_name, client, top_k=50):
    """Calculate suggested labels using OpenSearch KNN search."""
    df_labeled = df[df['label'] != ""]
    df_unlabeled = df[df['label'] == ""]

    label_embeddings = {}
    for label in df_labeled['label'].unique():
        embeddings = df_labeled[df_labeled['label'] ==
                                label]['full_embedding'].head(3).tolist()
        label_embeddings[label] = embeddings

    aggregated_results = defaultdict(list)

    for label, embeddings_label in label_embeddings.items():
        responses = search_similar_embeddings(
            client, index_name, embeddings_label, top_k)
        for response in responses:
            for result in response:
                aggregated_results[label].append(
                    result['_source']['document_id'])

    new_dataframe = pd.DataFrame()

    for label, doc_ids in aggregated_results.items():
        filtered_df = df_unlabeled[df_unlabeled['document_id'].isin(
            doc_ids)].copy()
        filtered_df['suggested_label'] = label
        new_dataframe = pd.concat(
            [new_dataframe, filtered_df], ignore_index=True)

    return new_dataframe


def calculate_suggested_label_neural(df, n_similares, threshold, es_client, index_name):
    """Calculate suggested labels using neural search with OpenSearch."""
    df_labeled = df[df['label'] != ""]
    df_unlabeled = df[df['label'] == ""]

    suggested_labels = []
    mean_top_similarities = []
    most_common_counts = []

    for _, row in df_unlabeled.iterrows():
        embedding_unlabeled = row['full_embedding']

        query = {
            "size": n_similares,
            "query": {
                "knn": {
                    "full_embedding": {
                        "vector": embedding_unlabeled,
                        "k": n_similares
                    }
                }
            }
        }

        response = es_client.search(index=index_name, body=query)

        similar_labels = [hit['_source']['label']
                          for hit in response['hits']['hits']]
        similarities = [hit['_score'] for hit in response['hits']['hits']]

        most_common_label = Counter(similar_labels).most_common(1)[0][0]
        suggested_labels.append(most_common_label)
        most_common_count = Counter(similar_labels)[most_common_label]
        most_common_counts.append(most_common_count)

        mean_similarity = np.mean(similarities)
        mean_top_similarities.append(mean_similarity)

    df_unlabeled['mean_similarity'] = mean_top_similarities
    df_unlabeled['count'] = most_common_counts
    df_unlabeled['suggested_label'] = suggested_labels
    df_unlabeled = df_unlabeled[df_unlabeled['mean_similarity'] > threshold].sort_values(
        by='mean_similarity', ascending=False)

    df.update(df_unlabeled)

    return df_unlabeled


def fetch_documents_by_query(es, index, search_string):
    """Fetch documents from OpenSearch by matching a query string."""
    scroll_timeout = "2m"
    size = 1000
    all_documents = []

    query = {
        "query": {
            "match": {
                "document_content": search_string
            }
        },
        "_source": [
            "document_content", "document_id", "label", "full_embedding", "process_type"
        ]
    }

    response = es.search(index=index, body=query,
                         scroll=scroll_timeout, size=size)
    scroll_id = response['_scroll_id']
    documents = response['hits']['hits']
    all_documents.extend(documents)

    while len(documents) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
        documents = response['hits']['hits']
        all_documents.extend(documents)

    return all_documents


def fetch_documents_by_type(es, index, search_string):
    """Fetch documents from OpenSearch by matching a type."""
    scroll_timeout = "2m"
    size = 1000
    all_documents = []

    query = {
        "query": {
            "match": {
                "type": search_string
            }
        },
        "_source": [
            "document_content", "document_id", "label", "full_embedding", "process_type"
        ]
    }

    response = es.search(index=index, body=query,
                         scroll=scroll_timeout, size=size)
    scroll_id = response['_scroll_id']
    documents = response['hits']['hits']
    all_documents.extend(documents)

    while len(documents) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
        documents = response['hits']['hits']
        all_documents.extend(documents)

    return all_documents


def calculate_similarity(df, n_clusters):
    """Calculate similarity and assign clusters to the DataFrame."""
    embeddings = np.array(df["embedding"].tolist())

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)
    df['cluster'] = labels

    return df
