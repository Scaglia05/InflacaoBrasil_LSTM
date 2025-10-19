import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def gerar_previsao(df, modelo, modelo_tipo="rf", n_lags=12, HORIZONTE=24, scaler=None):
    """
    Gera previsão futura.
    modelo_tipo: "rf", "linear" ou "lstm"
    HORIZONTE: meses à frente
    scaler: necessário apenas para LSTM
    """
    df = df.copy()
    
    # Criar lags
    for lag in range(1, n_lags+1):
        if f'lag_{lag}' not in df.columns:
            df[f'lag_{lag}'] = df['valor'].shift(lag)
    df = df.dropna()
    
    future_dates = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=HORIZONTE, freq='MS')
    future_preds = []
    
    RF_LAGS = [1,3,6,12]
    if modelo_tipo in ["rf", "linear"]:
        last_known = df.iloc[-1][[f'lag_{lag}' for lag in RF_LAGS]].values
        for _ in range(HORIZONTE):
            pred = modelo.predict(last_known.reshape(1,-1))[0]
            future_preds.append(pred)
            last_known = np.roll(last_known, -1)
            last_known[-1] = pred
    
    elif modelo_tipo == "lstm":
        last_known = df['valor'].values[-n_lags:]
        for _ in range(HORIZONTE):
            scaled_input = scaler.transform(last_known.reshape(-1,1))
            pred_scaled = modelo.predict(scaled_input.reshape(1, n_lags,1), verbose=0)
            pred = scaler.inverse_transform(pred_scaled)[0,0]
            future_preds.append(pred)
            last_known = np.roll(last_known, -1)
            last_known[-1] = pred
    
    future_df = pd.DataFrame(future_preds, index=future_dates, columns=['valor'])
    
    # Plot
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['valor'], label='Histórico', color='blue')
    plt.plot(future_df.index, future_df['valor'], label=f'Previsão +{HORIZONTE//12} anos', color='red', linestyle='--')
    plt.title("IPCA Mensal: Histórico e Previsão")
    plt.xlabel("Ano")
    plt.ylabel("Inflação (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    out = Path("data")
    out.mkdir(exist_ok=True)
    future_df.to_csv(out/"ipca_previsao_ml.csv")
    print(f"Previsão +{HORIZONTE//12} anos salva em data/ipca_previsao_ml.csv")
    
    return future_df
