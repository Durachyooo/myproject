import gradio as gr
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. Загружаем модель и вспомогательные объекты
# -----------------------------
model = joblib.load("hdd_mvp_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
trained_features = joblib.load("trained_features.pkl")  # список колонок модели

# Кодировка категорий (пример для HDD-MVP)
machine_mapping = {'L': 0, 'M': 1, 'H': 2}

# -----------------------------
# 2. Функция предсказания
# -----------------------------
def predict_mvp(csv_file):
    # 1️⃣ Читаем CSV
    df_raw = pd.read_csv(csv_file.name)
    
    # 2️⃣ Убираем колонку 'failure', если есть
    if 'failure' in df_raw.columns:
        df_raw = df_raw.drop('failure', axis=1)
    
    # 3️⃣ Выравниваем колонки: оставляем только нужные
    df = df_raw.copy()
    for col in trained_features:
        if col not in df.columns:
            df[col] = np.nan  # заполняем NaN, имputer справится
    df = df[trained_features]
    
    # 4️⃣ Импутация пропусков
    df_imputed = imputer.transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=trained_features)
    
    # 5️⃣ Масштабирование
    df_scaled = scaler.transform(df_imputed)
    df_scaled = pd.DataFrame(df_scaled, columns=trained_features)
    
    # 6️⃣ Кодируем категориальный признак, если есть
    if 'Machine_Type' in df_scaled.columns:
        df_scaled['Machine_Type'] = df_scaled['Machine_Type'].map(machine_mapping)
    
    # 7️⃣ Предсказание вероятности отказа
    df_raw['failure_prob'] = model.predict_proba(df_scaled)[:, 1]
    
    # 8️⃣ Форматируем результат для пользователя
    output_df = df_raw[['serial_number', 'model', 'capacity_bytes', 'failure_prob']].copy()
    output_df['Вероятность отказа'] = (output_df['failure_prob'] * 100).round(2).astype(str) + '%'
    output_df = output_df.rename(columns={
        'serial_number': 'Серийный номер диска',
        'model': 'Модель диска',
        'capacity_bytes': 'Емкость (байты)'
    })
    output_df = output_df[['Серийный номер диска', 'Модель диска', 'Емкость (байты)', 'Вероятность отказа']]
    
    # Сортируем по вероятности отказа
    output_df = output_df.sort_values(by='Вероятность отказа', ascending=False)
    
    return output_df.head(10)

# -----------------------------
# 3. Gradio интерфейс
# -----------------------------
iface = gr.Interface(
    fn=predict_mvp,
    inputs=gr.File(file_types=['.csv']),
    outputs="dataframe"
)

# -----------------------------
# 4. Запуск для Render
# -----------------------------
iface.launch(server_name="0.0.0.0", server_port=8080)
