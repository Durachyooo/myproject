import gradio as gr
import pandas as pd
import joblib

model = joblib.load("hdd_mvp_model.pkl")

def predict_mvp(csv_file):
    df = pd.read_csv(csv_file.name)
    # если колонка 'failure' есть, убираем
    if 'failure' in df.columns:
        df = df.drop('failure', axis=1)
    
    df['failure_prob'] = model.predict_proba(df)[:, 1]
    
    df['Вероятность отказа'] = (df['failure_prob']*100).round(2).astype(str) + '%'
    
    return df.head(10)

gr.Interface(fn=predict_mvp, inputs=gr.File(file_types=['.csv']), outputs="dataframe")\
    .launch(server_name="0.0.0.0", server_port=8080)
