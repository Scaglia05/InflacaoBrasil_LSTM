import requests
import pandas as pd
from pathlib import Path

# Criar pasta data
Path("data").mkdir(exist_ok=True)

# Série IPCA mensal (%), código 433 no SGS
series_id = 433
start = "01/01/2000"
end   = "31/12/2025"
url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json&dataInicial={start}&dataFinal={end}"

r = requests.get(url, timeout=30)
r.raise_for_status()
data = pd.DataFrame(r.json())
data['data'] = pd.to_datetime(data['data'], dayfirst=True)
data['valor'] = pd.to_numeric(data['valor'].str.replace(',', '.'))
data = data.sort_values('data').set_index('data')

# Criar índice contínuo e preencher faltantes
full_index = pd.date_range(start='2000-01-01', end=data.index.max(), freq='MS')
data = data.reindex(full_index)
data['valor'] = data['valor'].fillna(method='ffill')

# Salvar CSV
data.to_csv("data/ipca_continuo.csv")
