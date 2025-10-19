import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Carregar série contínua do IPCA
data = pd.read_csv("data/ipca_continuo.csv", index_col=0, parse_dates=True)

# Plot autocorrelação (ACF)
plt.figure(figsize=(12,4))
plot_acf(data['valor'], lags=36)
plt.title("Autocorrelação do IPCA Mensal (36 lags)")
plt.xlabel("Defasagens (meses)")
plt.ylabel("Autocorrelação")
plt.grid(True)
plt.savefig("data/acf_ipca.png")
plt.show()

# Plot parcial de autocorrelação (PACF) para análise adicional
plt.figure(figsize=(12,4))
plot_pacf(data['valor'], lags=36, method='ywm')
plt.title("Autocorrelação Parcial do IPCA Mensal (36 lags)")
plt.xlabel("Defasagens (meses)")
plt.ylabel("Autocorrelação Parcial")
plt.grid(True)
plt.savefig("data/pacf_ipca.png")
plt.show()
