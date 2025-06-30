# Импорт библиотек для взаимодействия с векторной БД Qdran
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import uuid

# Библиотеки для работы с датафреймами
import pandas as pd
import numpy as np

############
async def create_qdrant():
    # Шаг-1. Загружаем предобратонный DataFrame из файла pkl
    df_loaded = pd.read_pickle('src/my_dataframe.pkl')
    
    # Шаг-2. Загрузка данных в БД Qdran 
    vectors = df_loaded['embeddings_questions']
    vectors = vectors.tolist() # переводим siries в массив  , vectors[0].shape[0] # размерность
    data_dict = df_loaded.to_dict(orient='records')  # Преобразование DF в словарь
    # Запросы к Qdran
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "questions_chat"
    # Создаем коллекцию, если её нет
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vectors[0].shape[0], distance=Distance.COSINE),  # size=vectors.shape[1]
        )

        #Готовим точки для загрузки
        points = [
            PointStruct(
                id=str(uuid.uuid4()),  # уникальный ID для каждой записи
                vector=vector.tolist(),  # 
                payload={
                    "question": item["question"],
                    "preprocessed_question": item["preproc_question"],
                    "answer": item["answer"],
                    "date_created": item["date"]
                }
            )
            for vector, item in zip(vectors, data_dict)
        ]
        # Отправляем данные в Qdrant
        client.upsert(collection_name=collection_name, wait=True, points=points)

    except: 
        print("Коллекция уже есть в Qdran")
