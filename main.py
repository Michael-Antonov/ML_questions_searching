# Импорт fastapi библиотек 
from fastapi import FastAPI, Depends
from fastapi import APIRouter

# Импортер эмбендинг FastText 
import fasttext

# Импорт библиотек для автоматического переподъема приложения
from contextlib import asynccontextmanager

# Импорт самопиисная функция на поднятие базы
from src.database import create_qdrant
# Импорт созданных методов API 
from src.api_router import router as router_questions


# Правило для включения/выключения приложения FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_qdrant()
    print("БД готова к работе")
    yield
    print("Выключение")

app = FastAPI(lifespan=lifespan)

# Загрузка энкодера FastText. Логируем в терминале этапы загрузки 
print("начинаем загрузку энкодера")
path  = "src/cc.ru.300.bin" #"C:/Users/anton/OneDrive/Рабочий стол/3_case_AI_marafon/new_model/cc.ru.300.bin"
model_encoder = fasttext.load_model(path) 
print("энкодер готов")
print("приложение готово к работе")
########################################################################################################################

# Импорт API-методов
app.include_router(router=router_questions, prefix="/api/v1",tags=['questions'])


