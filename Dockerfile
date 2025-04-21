FROM python:3.6-slim

WORKDIR /app

# Встановлення необхідних системних залежностей, включаючи LLVM
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    llvm-9 \
    llvm-9-dev \
    llvm-9-runtime \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Створюємо символічне посилання для llvm-config
RUN ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config

# Встановлення залежностей Python
RUN pip install --no-cache-dir \
    dash==1.20.0 \
    dash-bootstrap-components==0.12.0 \
    plotly==4.14.3 \
    pandas==1.1.5 \
    numpy==1.19.5 \
    scipy==1.5.4 \
    networkx==2.5.1 \
    scikit-learn==0.24.2 \
    openpyxl==3.0.7 \
    matplotlib==3.3.4

# Встановлюємо numba окремо, після всіх інших пакетів
RUN pip install --no-cache-dir numba==0.53.1

# Копіювання файлів
COPY . /app/

# Створення директорії для зберігання даних
RUN mkdir -p /app/saved_data

# Відкриття порту для Dash-додатку
EXPOSE 8050

# Запуск додатку
CMD ["python", "FA_all.py"]