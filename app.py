import gradio as gr
import pandas as pd
import joblib

# -----------------------------
# 1. Загружаем обученную модель sklearn
# -----------------------------
model = joblib.load("hdd_mvp_model.pkl")

# -----------------------------
# 2. Функция предсказания
# -----------------------------
def predict_mvp(csv_file):
    # Читаем CSV
    df = pd.read_csv(csv_file.name)
    
    # Убираем колонку 'failure', если есть
    if 'failure' in df.columns:
        df = df.drop('failure', axis=1)
    
    # Предсказание вероятности отказа
    df['failure_prob'] = model.predict_proba(df)[:, 1]
    
    # Форматируем колонку с вероятностью
    df['Вероятность отказа'] = (df['failure_prob'] * 100).round(2).astype(str) + '%'
    
    # Возвращаем только первые 10 строк
    return df.head(10)

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
