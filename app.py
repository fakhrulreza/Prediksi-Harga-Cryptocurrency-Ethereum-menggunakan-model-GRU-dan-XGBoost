import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# --- Konfigurasi Streamlit ---
st.set_page_config(layout="wide", page_title="Prediksi Harga Ethereum")
st.title("💰 Sistem Prediksi Harga Ethereum (ETH-USD)")
st.write("Analisis data historis dan prediksi harga ethereum menggunakan model Machine Learning.")

# --- Fungsi untuk Mengumpulkan Data ---
@st.cache_data
def load_eth_data(start_date='2021-01-01', end_date=date.today().strftime('%Y-%m-%d')):
    """
    Mengunduh data historis harga Ethereum dari Yahoo Finance.
    Data di-cache untuk menghindari pengunduhan berulang.
    """
    df = yf.download('ETH-USD', start=start_date, end=end_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[required_cols]
    df.dropna(inplace=True)
    
    return df

# --- MODIFIED: Logika Prediksi dengan XGBoost Baru ---
@st.cache_resource
def load_xgboost_model(model_path):
    """
    Memuat model XGBoost dari path yang diberikan.
    Model di-cache untuk menghindari pemuatan berulang.
    """
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(model_path)
    return xgb_model

def create_features(df):
    """
    Melakukan rekayasa fitur pada DataFrame untuk model XGBoost sesuai notebook.
    """
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year

    for lag in range(1, 8):
        df[f'lag_close_{lag}'] = df['Close'].shift(lag)

    df['rolling_mean_7'] = df['Close'].shift(1).rolling(window=7).mean()
    df['rolling_mean_30'] = df['Close'].shift(1).rolling(window=30).mean()

    return df

def predict_xgboost_days(model, full_data, target_days=15):
    """
    Membuat prediksi untuk beberapa hari ke depan menggunakan model XGBoost.
    """
    FEATURES_TO_USE = [
        'day_of_week', 'day_of_year', 'month', 'year',
        'lag_close_1', 'lag_close_2', 'lag_close_3', 'lag_close_4',
        'lag_close_5', 'lag_close_6', 'lag_close_7',
        'rolling_mean_7', 'rolling_mean_30'
    ]

    future_predictions = []
    last_date = full_data.index[-1]

    # Ambil data yang cukup untuk membuat fitur (lag dan rolling window)
    data_for_features = full_data.iloc[-(30 + target_days):].copy()

    for i in range(1, target_days + 1):
        # Buat fitur untuk data yang ada
        features_for_prediction = create_features(data_for_features)
        
        # Ambil baris terakhir yang fiturnya lengkap untuk prediksi
        input_features = features_for_prediction[FEATURES_TO_USE].iloc[-(target_days - i + 1)]

        # Buat DataFrame dari fitur input
        prediction_df = pd.DataFrame(input_features).T
        
        # FIX: Ganti nama kolom DataFrame agar sesuai dengan nama fitur yang diharapkan model.
        # Ini mengatasi masalah ketidakcocokan nama fitur (misalnya, spasi di akhir).
        try:
            prediction_df.columns = model.get_booster().feature_names
        except Exception:
            st.warning("Tidak dapat mengubah nama kolom untuk prediksi XGBoost. Melanjutkan dengan nama default.")

        # Lakukan prediksi
        prediction = model.predict(prediction_df)[0]

        # Simpan hasil prediksi
        future_date = last_date + timedelta(days=i)
        future_predictions.append({'Tanggal': future_date, 'Prediksi Harga (USD)': prediction})
    
    return pd.DataFrame(future_predictions)


# --- Prediksi dengan GRU (Tidak Diubah) ---
@st.cache_resource
def load_gru_assets(model_path, scaler_path):
    """
    Memuat model GRU dan MinMaxScaler dari path yang diberikan.
    """
    gru_model = load_model(model_path)
    gru_scaler = joblib.load(scaler_path)
    return gru_model, gru_scaler

def prepare_gru_data(data, scaler_gru, sequence_length=30):
    """
    Mempersiapkan data untuk model GRU.
    """
    dataset = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler_gru.transform(dataset)

    sequences = []
    labels = []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])
        labels.append(scaled_data[i + sequence_length, 0])
    
    X_seq = np.array(sequences)
    y_seq = np.array(labels)
    
    X_seq = np.reshape(X_seq, (X_seq.shape[0], X_seq.shape[1], 1))
    
    return X_seq, y_seq

def predict_gru_days(model, scaler, last_sequence, sequence_length):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(15): # GRU tetap memprediksi 7 hari
        next_step_pred_scaled = model.predict(current_sequence.reshape(1, sequence_length, 1), verbose=0)
        future_predictions.append(scaler.inverse_transform(next_step_pred_scaled)[0, 0])
        current_sequence = np.append(current_sequence[1:], next_step_pred_scaled, axis=0)
        
    return future_predictions


# --- Logika Utama Aplikasi Streamlit ---

# --- Sidebar Tampilan Baru ---
with st.sidebar:
    st.markdown("## ⚙️ Panel Kontrol")
    st.markdown("Gunakan panel ini untuk memilih model dan memulai prediksi harga Ethereum.")
    st.markdown("---")

    with st.container(border=True):
        st.markdown("##### 🤖 Pilih Model Prediksi")
        model_choice = st.radio(
            "Model yang tersedia:",
            ("GRU", "XGBoost"),
            captions=["Prediksi 15 hari", "Prediksi 15 hari"], # Caption diubah
            label_visibility="collapsed"
        )

    st.markdown("") # Spasi
    predict_button = st.button("🚀 Buat Prediksi", use_container_width=True, type="primary")

    st.markdown("---")
    with st.expander("Panduan Penggunaan"):
        st.write("""
        1.  **Pilih Model:** Gunakan pilihan di atas untuk memilih antara model GRU atau XGBoost.
        2.  **Buat Prediksi:** Klik tombol "Buat Prediksi" untuk melihat hasilnya.
        3.  **Analisis Hasil:** Lihat grafik dan tabel prediksi yang muncul di halaman utama.
        """)


with st.spinner("Mengunduh data Ethereum..."):
    eth_data = load_eth_data()

if eth_data.empty:
    st.error("❌ Tidak dapat mengunduh data Ethereum. Periksa koneksi internet Anda atau coba lagi nanti.")
else:
    st.success("✅ Data Ethereum berhasil diunduh!")

    st.subheader("📊 Metrik Utama")
    price_today = eth_data['Close'].iloc[-1]
    price_yesterday = eth_data['Close'].iloc[-2]
    price_change = price_today - price_yesterday
    price_change_pct = (price_change / price_yesterday) * 100
    volume_today = eth_data['Volume'].iloc[-1]
    high_today = eth_data['High'].iloc[-1]
    low_today = eth_data['Low'].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Harga Terakhir (USD)", f"${price_today:,.2f}", f"{price_change:,.2f} ({price_change_pct:.2f}%)")
    col2.metric("Tertinggi 24j", f"${high_today:,.2f}")
    col3.metric("Terendah 24j", f"${low_today:,.2f}")
    col4.metric("Volume 24j", f"{volume_today:,.0f}")
    
    st.markdown("---")

    future_predictions_df = pd.DataFrame()

    if predict_button:
        st.subheader("📈 Hasil Prediksi")

        if model_choice == "XGBoost":
            model_path = "models/xgbost_model.json" # Path model disesuaikan
            if os.path.exists(model_path):
                with st.spinner("Memuat model XGBoost dan membuat prediksi..."):
                    xgb_model = load_xgboost_model(model_path)
                    future_predictions_df = predict_xgboost_days(xgb_model, eth_data, target_days=15)
            else:
                st.error(f"❌ File model `{model_path}` tidak ditemukan.")

        elif model_choice == "GRU":
            model_path = "models/MODEL_GRU.h5"
            scaler_path = "models/scaler.pkl"
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with st.spinner("Memuat model GRU dan membuat prediksi..."):
                    gru_model, gru_scaler = load_gru_assets(model_path, scaler_path)
                    sequence_length = 30
                    X_gru, y_gru = prepare_gru_data(eth_data, gru_scaler, sequence_length)
                    if X_gru.shape[0] == 0:
                        st.error(f"❌ Data tidak cukup untuk membuat sekuens GRU.")
                    else:
                        last_sequence = X_gru[-1]
                        predictions = predict_gru_days(gru_model, gru_scaler, last_sequence, sequence_length)
                        future_dates = [eth_data.index[-1] + timedelta(days=i) for i in range(1, len(predictions) + 1)]
                        future_predictions_df = pd.DataFrame({
                            "Tanggal": future_dates,
                            "Prediksi Harga (USD)": predictions
                        })
            else:
                st.error(f"❌ File model `{model_path}` atau scaler `{scaler_path}` tidak ditemukan.")

        if not future_predictions_df.empty:
            pred_price_next_day = future_predictions_df['Prediksi Harga (USD)'].iloc[0]
            st.markdown(f"#### Prediksi harga untuk hari berikutnya: **${float(pred_price_next_day):,.2f}**")
            
            st.subheader("📈 Analisis Grafik & Prediksi Harga")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=eth_data.index, open=eth_data['Open'], high=eth_data['High'], low=eth_data['Low'], close=eth_data['Close'], name='Harga Historis'))
            
            # Plot prediksi masa depan
            fig.add_trace(go.Scatter(
                x=future_predictions_df['Tanggal'], 
                y=future_predictions_df['Prediksi Harga (USD)'], 
                mode='lines+markers', 
                name=f"Prediksi {len(future_predictions_df)} Hari ke Depan", 
                line=dict(color='gold', width=2)
            ))
            
            fig.update_layout(
                title_text=f"Analisis Harga Ethereum & Prediksi Model {model_choice}", 
                template="plotly_dark", 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), 
                yaxis_title="Harga (USD)", 
                xaxis_title="Tanggal", 
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"📄 Tabel Prediksi")
            display_df = future_predictions_df.copy()
            display_df['Prediksi Harga (USD)'] = display_df['Prediksi Harga (USD)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df.set_index('Tanggal'), use_container_width=True)
            
            csv = future_predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Prediksi (CSV)", 
                data=csv, 
                file_name=f"ethereum_prediction_{len(future_predictions_df)}days_{model_choice}.csv", 
                mime='text/csv'
            )

    st.markdown("---")

    with st.expander("ℹ️ Tentang Model yang Digunakan"):
        st.write("""
        ##### GRU (Gated Recurrent Unit)
        Model GRU adalah jenis Jaringan Saraf Tiruan (Neural Network) yang sangat baik dalam memahami pola dari data sekuensial seperti data time series harga. Kelebihannya adalah kemampuannya "mengingat" informasi dari masa lalu untuk membuat prediksi yang lebih akurat di masa depan. Model ini dapat memproyeksikan tren harga untuk beberapa hari ke depan.

        ##### XGBoost (Extreme Gradient Boosting)
        XGBoost adalah model berbasis *decision tree* yang sangat kuat dan cepat. Model ini bekerja dengan membuat banyak *decision tree* sederhana dan menggabungkan hasilnya. Untuk prediksi harga, XGBoost melihat kondisi pasar pada hari ini (berdasarkan fitur seperti *moving average*, harga sebelumnya, dan komponen waktu) untuk memprediksi harga beberapa hari ke depan.
        """)

    with st.expander("Tampilkan Data Historis"):
        st.dataframe(eth_data)
    
    st.info("⚠️ **Penting:** Prediksi yang dihasilkan oleh model ini adalah murni berdasarkan data historis dan tidak boleh dianggap sebagai saran finansial. Selalu lakukan riset Anda sendiri sebelum membuat keputusan investasi.")
