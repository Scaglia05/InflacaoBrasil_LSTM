import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Carregar série contínua
data = pd.read_csv("data/ipca_continuo.csv", index_col=0, parse_dates=True)

# --- Plot série histórica ---
plt.figure(figsize=(12,5))
plt.plot(data.index, data['valor'], label='IPCA Mensal', color='blue')
plt.title("Série de IPCA Mensal (2000 até último disponível)")
plt.xlabel("Ano")
plt.ylabel("Inflação (%)")
plt.grid(True)
plt.legend()
plt.show()

# --- Decomposição sazonal ---
decomp = seasonal_decompose(data['valor'], model='additive', period=12)  # sazonalidade anual

plt.figure(figsize=(12,8))
decomp.plot()
plt.suptitle("Decomposição Aditiva do IPCA Mensal", fontsize=16)
plt.show()

# --- Estatísticas descritivas ---
print("Média mensal:", round(data['valor'].mean(), 2))
print("Desvio padrão:", round(data['valor'].std(), 2))
print("Máximo:", data['valor'].max())
print("Mínimo:", data['valor'].min())

# --- Preparação opcional para LSTM ---
# Normalizar valores para LSTM
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data['valor_scaled'] = scaler.fit_transform(data[['valor']])
