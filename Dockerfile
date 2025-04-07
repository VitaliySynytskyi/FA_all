FROM python:3.6-slim

WORKDIR /app

# Встановлення необхідних системних залежностей
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копіювання файлів
COPY . /app/

# Створення директорії для зберігання даних
RUN mkdir -p /app/saved_data

# Встановлення залежностей Python
RUN pip install --no-cache-dir \
    dash==1.20.0 \
    dash-bootstrap-components==0.12.0 \
    plotly==4.14.3 \
    pandas==1.1.5 \
    numpy==1.19.5 \
    numba==0.53.1 \
    scipy==1.5.4 \
    networkx==2.5.1 \
    scikit-learn==0.24.2 \
    openpyxl==3.0.7 \
    matplotlib==3.3.4

# Відкриття порту для Dash-додатку
EXPOSE 8050

# Запуск додатку
CMD ["python", "FA_all.py"]