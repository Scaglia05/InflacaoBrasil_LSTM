from pathlib import Path
import pandas as pd
import numpy as np
import requests
from modelos import treinar_rf, treinar_linear, treinar_lstm
from AvaliarModelo import avaliar_modelo
from PrevisaoFinal import gerar_previsao

# --- Criar pasta data se não existir ---
Path("data").mkdir(exist_ok=True)

# --- Baixar CSV se não existir ---
csv_path = Path("data/ipca_continuo.csv")
if not csv_path.exists():
    print("CSV não encontrado. Baixando dados do SGS...")
    series_id = 433
    start = "01/01/2000"
    end = "31/12/2025"
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json&dataInicial={start}&dataFinal={end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = pd.DataFrame(r.json())
    data['data'] = pd.to_datetime(data['data'], dayfirst=True)
    data['valor'] = pd.to_numeric(data['valor'].str.replace(',', '.'))
    data = data.sort_values('data').set_index('data')
    full_index = pd.date_range(start='2000-01-01', end=data.index.max(), freq='MS')
    data = data.reindex(full_index)
    data['valor'] = data['valor'].fillna(method='ffill')
    data.to_csv(csv_path)
    print("CSV criado em:", csv_path)

# --- Carregar dados ---
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

HORIZONTE = 24  # 2 anos à frente
n_lags = 12

# --- Random Forest ---
RF_LAGS = [1, 3, 6, 12]  # lags usados no treinamento RF
rf_model, X_train_rf, y_train_rf = treinar_rf(df, ate_ano='2020-12-01')
val = df['2021-01-01':'2024-12-01']
X_val_rf = val[[f'lag_{lag}' for lag in [1,3,6,12]]]
y_val = val['valor']
y_pred_rf = rf_model.predict(X_val_rf)
avaliar_modelo(y_val, y_pred_rf, nome_modelo="Random Forest (2021-2024)")
rf_model_full, _, _ = treinar_rf(df, ate_ano='2024-12-01')
future_rf = gerar_previsao(df, rf_model_full, modelo_tipo="rf", n_lags=len(RF_LAGS), HORIZONTE=HORIZONTE)

# --- Regressão Linear ---
lr_model, X_train_lr, y_train_lr = treinar_linear(df, ate_ano='2020-12-01')
X_val_lr = val[[f'lag_{lag}' for lag in [1,3,6,12]]]
y_pred_lr = lr_model.predict(X_val_lr)
avaliar_modelo(y_val, y_pred_lr, nome_modelo="Linear Regression (2021-2024)")
lr_model_full, _, _ = treinar_linear(df, ate_ano='2024-12-01')
future_lr = gerar_previsao(df, lr_model_full, modelo_tipo="linear", n_lags=len(RF_LAGS), HORIZONTE=HORIZONTE)

# --- LSTM ---
lstm_model, X_train_lstm, y_train_lstm, scaler = treinar_lstm(df, ate_ano='2020-12-01', n_lags=n_lags, epochs=100, batch_size=16)
val_scaled = scaler.transform(val[['valor']])
X_val_lstm, y_val_lstm = [], []
for i in range(n_lags, len(val_scaled)):
    X_val_lstm.append(val_scaled[i-n_lags:i,0])
    y_val_lstm.append(val_scaled[i,0])
X_val_lstm = np.array(X_val_lstm).reshape(len(X_val_lstm), n_lags,1)
y_val_lstm = np.array(y_val_lstm)
y_pred_lstm_scaled = lstm_model.predict(X_val_lstm, verbose=0)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)
avaliar_modelo(val['valor'].iloc[n_lags:], y_pred_lstm, nome_modelo="LSTM (2021-2024)")
lstm_model_full, _, _, scaler_full = treinar_lstm(df, ate_ano='2024-12-01', n_lags=n_lags, epochs=100, batch_size=16)
future_lstm = gerar_previsao(df, lstm_model_full, modelo_tipo="lstm", scaler=scaler_full, n_lags=n_lags, HORIZONTE=HORIZONTE)
