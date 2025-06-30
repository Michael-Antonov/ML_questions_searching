# Импорт fastapi библиотек 
from fastapi import FastAPI, Depends, Body
from pydantic import BaseModel
# Библиотека для иницилизации роутов по API-методы
from fastapi import APIRouter
# Импорт библиотек для взаимодействия с векторной БД Qdran
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import uuid
# Библиотеки для работы с датафреймами
import pandas as pd
import numpy as np 
# Библиотека для работы со временем
from datetime import date
# библиотеки необходимые для функции, которая будет делать препроцессинг текста 
import re
import pymorphy3
import nltk
from nltk.corpus import stopwords

# Запросы к Qdran
client = QdrantClient(host="localhost", port=6333)
collection_name = "questions_chat"


router = APIRouter() #Инициализация клсса роутов

# Ф-1. Создание энкодера 
def get_fasttext_embedding(text, model):
    words = text.split()
    vectors = [model.get_word_vector(word) for word in words if word in model]
    if not vectors:
        return np.zeros(300) #model.get_dimension()
    return np.mean(vectors, axis=0)

# Ф-2. Базовый препроцессинг текста.
# Загрузка стоп-слов - загружаем единоразово
nltk.download('stopwords')
russian_stopwords = set(stopwords.words('russian'))
# Свой локальный набор стоп-слов
local_stopwords = ["здравствуйте", "добрый день", "добрый вечер", "доброе утро", "спасибо"] 
# Инициализация лемматизатора
morph = pymorphy3.MorphAnalyzer(lang='ru')

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление всех символов, кроме букв и пробелов
    text = re.sub(r'[^а-яё\s]', '', text)
    
    # Разделение на слова
    words = text.split()

    # Лемматизация и фильтрация стоп-слов
    processed_words = []
    for word in words:
        if word in russian_stopwords:
            continue
        if word in local_stopwords:
            continue 
        parsed_word = morph.parse(word)[0]
        lemma = parsed_word.normal_form
        processed_words.append(lemma)
    
    # Сборка обратно в строку
    return ' '.join(processed_words)

# Ф-3. Функция под метод для поиска дубликатов
def use_model_encoder(): # для импорта энкодинг модели FastText
    from src.main import model_encoder
    return model_encoder
#from test import test_query_vector # для тестовы вместо импорта model_encoder = FastText
def search_duplicate_in_qdrant(query):
    #cleaned_query = query #preprocess_text(query) # чистим и нормализируем текст 
    query_vector =  get_fasttext_embedding(query, use_model_encoder()) # векторизируем текст для поиска !
    #query_vector = test_query_vector # Для тестирования без импорта FastText испольщуем тестовый вектор !
    # Делаем запрос в Qdran для поиска наиболее похожих вопросов по алгоритму ANN. Метрика схожести - косинусная близость
    search_result = client.search ( 
    collection_name= "questions_chat",
    query_vector=query_vector,
    limit=3
    )
    
    #Собираем структуру ответа
    most_similiar_questions = []
    for hit in search_result:
        hit_object = {
            "question": hit.payload['question'],
            "preprocessed_question": hit.payload['preprocessed_question'],
            "answer": hit.payload['answer'],
            "date_created": hit.payload['date_created'],
            "similarity_score": hit.score
        }
        most_similiar_questions.append(hit_object)
    
    return most_similiar_questions

#Ф-4. Для сохранения нового вопроса в БД
def save_new_question_in_qdrant(Qdrant_client, Qdrant_collection_name, question, answer):
    
    vector = get_fasttext_embedding(question, use_model_encoder()) #model_encoder
    #vector = test_query_vector

    today = date.today()
    today_str = today.isoformat() # сохранение даты в формате YYYY-MM-DD

    preprocessed_question = preprocess_text(question)

    # Полезная нагрузка (payload)
    payload = {
        "question": f"{question}",
        "preprocessed_question": f"{preprocessed_question}",
        "answer": f"{answer}",
        "date_created": f"{today_str}"
    }

    # Создаем точку для Qdrant с уникальным id
    point = PointStruct(
        id=str(uuid.uuid4()), 
        vector=vector,
        payload=payload
    )

    # Добавляем точку в коллекцию Qdrant
    operation_result = Qdrant_client.upsert(
        collection_name=Qdrant_collection_name,
        wait=True,
        points=[point]
    )
    return operation_result

# Создание методов API 
# Метод-1. Поиск похожих вопросов - POST "/questions/searching"
class SearchQuestion(BaseModel): # Объявляем структуру класса SerchQuestion
    question: str

    model_config = {
        "json_schema_extra":{
            "question": "Добрый день. Получил уведомление о том, что с 12 числа кэшбэк по Халве увеличивается в 2 раза с подпиской «Халва. Десятка». И ссылка на информацию.Где можно найти понятное и подробное описание?"
        }
    }

class QuestionResponse(BaseModel): # Объявляем структуру класса SerchQuestion
    question: str
    preprocessed_question: str
    answer: str
    date_created: str
    similarity_score: float



@router.post("/questions/searching", 
             description = "Поиск похожиших вопросов",)

async def search_questions(
    question: SearchQuestion= Body(
        ...,
        examples=[
            {
                "question": "Добрый день. Получил уведомление о том, что с 12 числа кэшбэк по Халве увеличивается в 2 раза с подпиской «Халва. Десятка». И ссылка на информацию.Где можно найти понятное и подробное описание?"
            }
        ]
    )
):
    return {
        "most_similiar_questions": search_duplicate_in_qdrant(question.question)
    }

# Метод-2. Поиск похожих вопросов - POST "/questions"
class Question(BaseModel):
    question: str
    answer: str

    model_config = {
        "json_schema_extra":{
            "question": "Новый вопрос?",
            "answer": "Ответ на новый вопрос"
        }
    }

@router.post("/questions", 
             description = "Записать в Базу новый вопрос с готовым ответом")

async def post_questions(
    new_question: Question = Body(
        ...,
        examples=[
            {
                "question": "Новый вопрос?",
                "answer": "Ответ на новый вопрос" 
                }
            ]
    )
):
    qdrant_result = save_new_question_in_qdrant(client, collection_name, new_question.question, new_question.answer)
    return {
            'message': "Новый вопрос с ответом успешно добавлен",
            "qdrant_response": qdrant_result
        }