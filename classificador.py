import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import connect_opensearch, fetch_documents, create_dataframe, persist_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


    
def search_similar_vectors(es, index, embedding, top_n=100):
    query = {
        "size": top_n,
        "query": {
            "knn": {
                "embedding_completo": {
                    "vector": embedding,
                    "k": top_n
                }
            }
        }
    }
    response = es.search(index=index, body=query)
    return response['hits']['hits']

def classify_by_similarity(es, index, embedding, top_n=100):
    # Busca os vetores mais similares
    similar_docs = search_similar_vectors(es, index, embedding, top_n=top_n)
    
    # Filtra apenas os documentos que têm rótulos
    labeled_docs = [doc for doc in similar_docs if doc['_source'].get('label')]
    
    # Calcula as estatísticas
    if labeled_docs:
        labels = [doc['_source']['label'] for doc in labeled_docs]
        predominant_label = max(set(labels), key=labels.count)
        predominant_percentage = labels.count(predominant_label) / len(labeled_docs) * 100
        
        unlabeled_percentage = (len(similar_docs) - len(labeled_docs)) / len(similar_docs) * 100
        distances = [doc['_score'] for doc in similar_docs]  # Assume que _score representa similaridade
    else:
        predominant_label = "Nenhum"
        predominant_percentage = 0
        unlabeled_percentage = 100
        distances = []

    return predominant_label, predominant_percentage, unlabeled_percentage, distances

def app():
    st.title('Classificação de Processos')
    
    # Seleção de Modo: Treinamento, Classificação com Modelo ou Classificação por Similaridade
    mode = st.sidebar.selectbox("Escolha o modo de operação", ["Treinamento", "Classificação com Modelo"])

    if mode == "Treinamento":
        st.header("Treinamento de Modelos")
        train_test_ratio = st.sidebar.slider("Defina a porcentagem para teste", 10, 50, 20) / 100.0
        seed=st.sidebar.text_input("Seed", "50")
        if st.sidebar.button("Submeter"):
            HOST = "10.10.25.161"
            PORT = 9200
            opensearch_index = "classificador_dados_sensiveis"
            es = connect_opensearch(HOST, PORT)
            documents = fetch_documents(es, opensearch_index)
            df_new = create_dataframe(documents)
            df_labeled = df_new[df_new['label'].notnull() & (df_new['label'].str.strip() != '')]
            X_new = df_labeled['embedding_completo'].tolist()
            y = df_labeled['label'].tolist()
            X_new = np.array(X_new)
            y = np.array(y)
            indices = df_labeled.index
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                X_new, y_encoded, indices, test_size=train_test_ratio, random_state=int(seed), stratify=y_encoded
            )
            df_test = df_labeled.loc[test_indices]
            st.write("Treinando Rede Neural.")
            nn_clf = MLPClassifier(max_iter=10000)
            nn_clf.fit(X_train, y_train)

            joblib.dump(nn_clf, 'nn_model.pkl')
            joblib.dump(label_encoder, 'label_encoder.pkl')
            st.success("Modelo treinado e salvo com sucesso!")
            st.write("Avaliando modelo...")
            nn_predictions = nn_clf.predict(X_test)
            predictions = label_encoder.inverse_transform(nn_predictions)
            test_true_label = label_encoder.inverse_transform(y_test)
            evaluate_classifiers(test_true_label, predictions, df_test)

    elif mode == "Classificação com Modelo":
        st.header("Classificação de Instâncias Não Rotuladas com Modelos Treinados")
        if st.sidebar.button("Submeter"):
            # Carregar os modelos treinados e o LabelEncoder
            svm_clf = joblib.load('svm_model.pkl')
            rf_clf = joblib.load('rf_model.pkl')
            nn_clf = joblib.load('nn_model.pkl')
            label_encoder = joblib.load('label_encoder.pkl')  # Carrega o mesmo LabelEncoder usado no treinamento

            # Conectar ao OpenSearch e buscar documentos não rotulados
            HOST = "10.10.25.161"
            PORT = 9200
            INDEX = "siged-pj-similaridade-2024"
            es = connect_opensearch(HOST, PORT)

            # Carregar e armazenar dados no session_state
            if 'df_unlabeled' not in st.session_state:
                documents = fetch_documents(es, INDEX)
                df_new = create_dataframe(documents)
                df_unlabeled = df_new[df_new['label'].isnull() | (df_new['label'].str.strip() == '')]

                if df_unlabeled.empty:
                    st.warning("Não há instâncias não rotuladas para classificar.")
                    return

                X_unlabeled = df_unlabeled['embedding_completo'].tolist()
                X_unlabeled = np.array(X_unlabeled)

                # Fazer predições com cada classificador
                svm_predictions = svm_clf.predict(X_unlabeled)
                rf_predictions = rf_clf.predict(X_unlabeled)
                nn_predictions = nn_clf.predict(X_unlabeled)

                # Votação majoritária
                majority_vote_predictions_encoded = majority_vote(svm_predictions, rf_predictions, nn_predictions)

                # Decodificar predições para labels originais
                majority_vote_predictions = label_encoder.inverse_transform(majority_vote_predictions_encoded)

                st.session_state.df_unlabeled = df_unlabeled
                st.session_state.label_predictions = majority_vote_predictions
                st.session_state.distinct_labels = sorted(df_new['label'].dropna().unique().tolist())
                st.session_state.current_instance = 0

            current_instance = st.session_state.current_instance
            total_instances = len(st.session_state.df_unlabeled)

            # Exibir a instância atual
            row = st.session_state.df_unlabeled.iloc[current_instance]
            predicted_label = st.session_state.label_predictions[current_instance]

            st.header(f"Instância {current_instance + 1} de {total_instances}")
            st.text_area(f"Texto do Documento (ID: {row['id_documento']})", row['ds_documento_ocr'][:1000], height=150)
            st.write(f"Rótulo Sugerido: {predicted_label}")

            # Seleção de rótulo
            distinct_labels = st.session_state.distinct_labels
            selected_label = st.selectbox("Selecione um rótulo", distinct_labels, index=distinct_labels.index(predicted_label) if predicted_label in distinct_labels else 0)

            # Navegação entre as instâncias
            nav1, nav2, nav3 = st.columns([1, 1, 2])
            with nav1:
                if st.button("Anterior"):
                    st.session_state.current_instance = (current_instance - 1) % total_instances
            with nav2:
                if st.button("Próximo"):
                    st.session_state.current_instance = (current_instance + 1) % total_instances

            # Salvar rótulo confirmado
            if st.button("Confirmar Rótulo"):
                st.session_state.df_unlabeled.at[st.session_state.df_unlabeled.index[current_instance], 'label'] = selected_label
                persist_labels(es, INDEX, st.session_state.df_unlabeled.loc[[st.session_state.df_unlabeled.index[current_instance]]])
                st.success(f"Rótulo '{selected_label}' confirmado para o documento {row['id_documento']}")

    elif mode == "Classificação por Similaridade":
        st.header("Classificação de Instâncias Não Rotuladas por Similaridade")
        topk = st.sidebar.slider("Defina o valor top N", 3, 50, 5)
   
        # Conectar ao OpenSearch e buscar documentos não rotulados
        HOST = "10.10.25.161"
        PORT = 9200
        INDEX = "siged-pj-similaridade-2024"
        es = connect_opensearch(HOST, PORT)

        if 'df_unlabeled' not in st.session_state:
            documents = fetch_documents(es, INDEX)
            df_new = create_dataframe(documents)
            df_unlabeled = df_new[df_new['label'].isnull() | (df_new['label'].str.strip() == '')]

            if df_unlabeled.empty:
                st.warning("Não há instâncias não rotuladas para classificar.")
                return

            st.session_state.df_unlabeled = df_unlabeled
            st.session_state.current_instance = 0

        current_instance = st.session_state.current_instance
        total_instances = len(st.session_state.df_unlabeled)

        # Exibir a instância atual
        row = st.session_state.df_unlabeled.iloc[current_instance]
        embedding = row['embedding_completo']

        st.header(f"Instância {current_instance + 1} de {total_instances}")
        st.text_area(f"Texto do Documento (ID: {row['id_documento']})", row['ds_documento_ocr'][:1000], height=150)

        # Classificar por similaridade
        predominant_label, predominant_percentage, unlabeled_percentage, distances = classify_by_similarity(es, INDEX, embedding, top_n=topk)

        st.write(f"Total de documentos com rótulo: {100 - int(unlabeled_percentage)}")
        st.write(f"Rótulo predominante: {predominant_label} ({predominant_percentage:.2f}%)")
        st.write(f"Percentual de documentos sem rótulo: {unlabeled_percentage:.2f}%")

        if distances:
            st.write(f"Distância cosseno mínima entre os documentos: {min(distances):.4f}")
            st.write(f"Distância cosseno máxima entre os documentos: {max(distances):.4f}")

        # Navegação entre as instâncias
        nav1, nav2, nav3 = st.columns([1, 1, 2])
        with nav1:
            if st.button("Anterior"):
                st.session_state.current_instance = (current_instance - 1) % total_instances
        with nav2:
            if st.button("Próximo"):
                st.session_state.current_instance = (current_instance + 1) % total_instances

        # Salvar rótulo sugerido
        if st.button("Confirmar Rótulo Predominante"):
            st.session_state.df_unlabeled.at[st.session_state.df_unlabeled.index[current_instance], 'label'] = predominant_label
            persist_labels(es, INDEX, st.session_state.df_unlabeled.loc[[st.session_state.df_unlabeled.index[current_instance]]])
            st.success(f"Rótulo '{predominant_label}' confirmado para o documento {row['id_documento']}")


def evaluate_classifiers(y_test, svm_predictions,df_labeled):
    vote_accuracy = accuracy_score(y_test, svm_predictions)
    vote_precision = precision_score(y_test, svm_predictions, average='weighted')
    vote_recall = recall_score(y_test, svm_predictions, average='weighted')
    vote_f1 = f1_score(y_test, svm_predictions, average='weighted')
    vote_confusion = confusion_matrix(y_test, svm_predictions)

    st.write("Desempenho do Classificador de Votação Majoritária (SVM, RF, NN):")
    st.write(f"Acurácia: {vote_accuracy:.4f}")
    st.write(f"Precisão: {vote_precision:.4f}")
    st.write(f"Recall: {vote_recall:.4f}")
    st.write(f"F1 Score: {vote_f1:.4f}")
    st.write("Matriz de Confusão:")
    st.write(vote_confusion)
    st.write("\nRelatório de Classificação:")
    st.write(classification_report(y_test, svm_predictions))

    # Impressão dos textos ds_documento_ocr para erros de classificação
    incorrect_indices = np.where(svm_predictions != y_test)[0]
    st.write("Casos de erro na classificação:")
    for idx in incorrect_indices:
        st.write(f"True Label: {y_test[idx]}, Predicted Label: {svm_predictions[idx]}")
        st.write(f"Texto: {df_labeled.iloc[idx]['ds_documento_ocr']}")
        if st.button(f"Atualizar no OpenSearch - ID {df_labeled.iloc[idx]['id_documento']}", key=f"btn_{idx}"):
            response = update_opensearch(opensearch_index, id_documento, true_label, predicted_label)
            st.write(f"Documento atualizado no OpenSearch: {response}")

def update_opensearch(index, doc_id, true_label, predicted_label):
    client = OpenSearch(

        hosts=[{'host': '10.10.25.161', 'port': 9200}],  # Substitua pelo host do OpenSearch
        http_auth=('username', 'password')           # Substitua pelas credenciais do OpenSearch
    )
    
    body = {
        "doc": {
            "label": predicted_label
        }
    }
    
    response = client.update(index=index, id_documento=id_documento, body=body)
    return response

if __name__ == "__main__":
    app()
