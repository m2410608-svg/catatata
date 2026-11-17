# Классификация изображений Fashion‑MNIST

Пример проекта для практики "Архитектор ИИ".

Полный цикл ML-проекта: от исследовательского анализа данных и обучения модели до развёртывания в виде REST API.

---

## Задача

Классификация изображений одежды из датасета **Fashion‑MNIST** по 10 классам:


|Метка|Класс|
|:---|:---|
|0|T‑shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|

Датасет содержит 10 000 изображений размером 28×28 пикселей в оттенках серого. В рамках проекта используется подвыборка из 1000 объектов для быстрого обучения.

---

## Структура репозитория

```text
├── README.md                 # Описание проекта
├── requirements.txt          # Зависимости Python
├── Dockerfile                # Docker-образ для API
├── .gitignore                # Игнорируемые файлы
├── config.yaml               # Конфигурация (пути, гиперпараметры)
├── data/
|	└── images/           # Данные нашей задачи
│
├── src/                      
│   ├── __init__.py
│   ├── train.py              
│   ├── predict.py            
│   ├── api.py                # FastAPI-сервис
│   └── utils.py              
│
├── notebooks/                # Jupyter ноутбуки
│   └── Project_Zueva.ipynb   # EDA и пайплайн обучения и оценки модели
│
├── models/                   # Сериализованные модели
│   └── model.pkl             # Веса модели
│   └── features.pkl          # Признаки
│
├── tests/                    # Тесты
│   ├── __init__.py
│   └── test_api.py           # Тесты для API
│
└── scripts/                  # Скрипты для развёртывания
    ├── deploy.sh             # Сборка и запуск Docker
    └── test_request.sh       # Тестовые запросы к API
```

## Быстрый старт

### 1\. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2\. Запуск ноутбука с обучением модели

```bash
jupyter notebook notebooks/Project_Zueva.ipynb
```

После выполнения ноутбука модель сохранится в `models/model.pkl`.

### 3\. Запуск API

```bash
# Через uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Или через Docker
docker build -t fashion-mnist-api .
docker run -p 8000:8000 fashion-mnist-api
```

### 4\. Тестирование API

```bash
# Health check
curl http://localhost:8000/health

# Одиночное предсказание
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"data": [0.0, 0.0, ..., 0.0]}'   # 784 числа

# Пакетное предсказание
curl -X POST "http://localhost:8000/predict/batch" \
-H "Content-Type: application/json" \
-d '{"data": [[0.0]*784, [0.0]*784]}'
```

## Эндпоинты API


|Метод|Путь|Описание|
|:---|:---|:---|
|GET|/health|Проверка состояния сервиса|
|GET|/|Информация о сервисе|
|POST|/predict|Предсказание для одного объекта|
|POST|/predict/batch|Пакетное предсказание (до 1000)|

Документация Swagger: <http://localhost:8000/docs>

## **Модель**

* **Алгоритм:** Логистическая регрессия

* **Предобработка:** стандартизация пикселей через `StandardScaler`

* **Объём выборки:** подвыборка 1000 объектов (800 train / 200 test)

### **Метрики на тестовой выборке**

#|
||

**Метрики**

|

**Значения**

||
||

Accuracy

|

0\.8250

||
||

Macro Precision

|

0\.8060

||
||

Macro Recall

|

0\.8198

||
||

Macro F1-score

|

0\.8058

||
|#

Все метрики превышают установленные пороговые значения:

* Accuracy \> 0.75

* Macro F1 \> 0.70

* Macro Precision/Recall \> 0.70

---

## **Технологии**

* **Python** 3.11\+

* **scikit‑learn** — обучение модели

* **FastAPI** — REST API

* **Docker** — контейнеризация

* **Pytest** — тестирование API

* **NumPy / Pandas** — работа с данными

* **Matplotlib / Seaborn** — визуализация