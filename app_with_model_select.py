
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="TOC 예측 (모델 선택 포함)", layout="wide")
st.title("📈 TOC 예측 (CSV + 그래프 + 모델 선택)")

# 모델 선택 UI
model_option = st.selectbox("사용할 모델을 선택하세요", ["CNN", "CNN+LSTM"])
model_file = "model_a.h5" if model_option == "CNN" else "model_b.h5"

try:
    model = load_model(model_file, compile=False)
except:
    st.error(f"🚫 {model_file} 파일을 찾을 수 없습니다. 먼저 모델을 학습하고 저장해주세요.")
    st.stop()

scaler = joblib.load("scaler.pkl")

st.markdown("### 📥 CSV 업로드 또는 수동 입력")

tab1, tab2 = st.tabs(["📁 CSV 파일 입력", "⌨️ 수동 입력"])

def plot_prediction_graph(values):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(values)), values, marker='o', linestyle='-', color='blue')
    ax.set_title("입력된 TOC 시계열")
    ax.set_xlabel("시간 (10분 단위)")
    ax.set_ylabel("TOC (mg/L)")
    ax.grid(True)
    st.pyplot(fig)

# CSV 입력 탭
with tab1:
    uploaded_file = st.file_uploader("36행 × 5열(TOC, SS, Temp, TN, TP) CSV 파일을 업로드하세요", type=["csv"])
    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        st.dataframe(df_csv)
        if df_csv.shape == (36, 5):
            scaled_input = scaler.transform(df_csv).reshape(1, 36, 5)
            pred = model.predict(scaled_input)
            toc_scaled = float(pred[0][0])
            dummy = np.zeros((1, 5))
            dummy[0, 0] = toc_scaled
            toc_real = scaler.inverse_transform(dummy)[0, 0]
            st.success(f"[{model_option}] 예측된 TOC (정규화): **{toc_scaled:.3f}**, 📏 실제 단위: **{toc_real:.2f} mg/L**")
            plot_prediction_graph(df_csv['TOC'].values)
        else:
            st.error("⚠️ CSV 파일은 정확히 36행 × 5열이어야 합니다.")

# 수동 입력 탭
with tab2:
    input_data = []
    for i in range(36):
        cols = st.columns(5)
        toc = cols[0].number_input(f'TOC-{i+1}', value=5.0, key=f'toc_{i}')
        ss = cols[1].number_input(f'SS-{i+1}', value=20.0, key=f'ss_{i}')
        temp = cols[2].number_input(f'Temp-{i+1}', value=15.0, key=f'temp_{i}')
        tn = cols[3].number_input(f'TN-{i+1}', value=5.0, key=f'tn_{i}')
        tp = cols[4].number_input(f'TP-{i+1}', value=0.5, key=f'tp_{i}')
        input_data.append([toc, ss, temp, tn, tp])

    if st.button("🔮 수동 입력 예측 실행"):
        input_array = np.array(input_data)
        scaled_array = scaler.transform(input_array).reshape(1, 36, 5)
        prediction = model.predict(scaled_array)
        toc_scaled = float(prediction[0][0])
        dummy = np.zeros((1, 5))
        dummy[0, 0] = toc_scaled
        toc_real = scaler.inverse_transform(dummy)[0, 0]

        st.success(f"[{model_option}] 예측된 TOC (정규화): **{toc_scaled:.3f}**, 📏 실제 단위: **{toc_real:.2f} mg/L**")
        plot_prediction_graph([row[0] for row in input_data])
