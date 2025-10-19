import requests
import pandas as pd
from pathlib import Path

# Configurações
series_id = 433  # IPCA mensal
start = "01/01/2000"
end   = "31/12/2025"
out = Path("data")
out.mkdir(exist_ok=True)

# Requisição à API do SGS
url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json&dataInicial={start}&dataFinal={end}"
r = requests.get(url, timeout=30)
r.raise_for_status()
data = pd.DataFrame(r.json())

# Ajustes de formato
data['data'] = pd.to_datetime(data['data'], dayfirst=True)
data['valor'] = pd.to_numeric(data['valor'].str.replace(',', '.'))
data = data.sort_values('data').set_index('data')

# Criar índice mensal contínuo
full_index = pd.date_range(start='2000-01-01', end=data.index.max(), freq='MS')
data = data.reindex(full_index)

# Preencher valores faltantes
data['valor'] = data['valor'].fillna(method='ffill')

# Salvar CSV atualizado
data.to_csv(out/"ipca_continuo.csv")

# Checar primeiros e últimos valores
print(data.head(12))
print(data.tail(12))

# Meses originalmente sem dados
missing = data[data['valor'].isna()]
print(f"Meses sem dados originais: {len(missing)}")
print(missing)

# Para LSTM: normalização (opcional, pode ser aplicada no bloco de treino)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data['valor_scaled'] = scaler.fit_transform(data[['valor']])
