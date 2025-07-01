# ML_questions_searching
ML-проект для поиска похожих уже ранее заданных вопросов от сотрудников в треде с топ-менеджментом. Стек: Python FastAPI, FastText (embending model), Qdrant (ANN, Сosine similarity).

Для работы текущего образа проекта необходимо установить эмбенддинг-модель FastText cc.ru.300.bin и добавить ее в в папку src - https://fasttext.cc/docs/en/crawl-vectors.html?spm=a2ty_o01.29997173.0.0.440ec921IaCKOm

В папке scr должны находиться файлы: main.py, database.py, api_router.py, test.py, requirements.txt, my_dataframe.pkl(набор исходных данных с FastText эмбендингами тех сущностей, по которым нужно производить векторный поиск - их нужно подготовить самостоятельно)

Архитеткурная схема решения
![image](https://github.com/user-attachments/assets/878a04fc-ba8f-4f69-8b7a-8a512e51ccd4)
