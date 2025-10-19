import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def avaliar_modelo(y_real, y_pred, nome_modelo="Modelo", plot=True):
    """
    Avalia qualquer modelo (RF, Linear ou LSTM).
    - y_real: pd.Series ou np.array (valores reais)
    - y_pred: np.array (valores previstos)
    """
    # Ajustar se y_pred for 2D (como LSTM saída Dense)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
        y_pred = y_pred.flatten()
    
    # Garantir índices corretos
    if isinstance(y_real, pd.Series):
        index = y_real.index
    else:
        index = pd.RangeIndex(start=0, stop=len(y_real))
    
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    
    print(f"=== Avaliação: {nome_modelo} ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("===============================")
    
    # Salvar comparação
    df = pd.DataFrame({"Real": y_real, "Previsto": y_pred}, index=index)
    df.to_csv("data/comparacao_modelo.csv")
    
    # Plot
    if plot:
        plt.figure(figsize=(12,5))
        plt.plot(index, y_real, label='Real', color='blue')
        plt.plot(index, y_pred, label='Previsto', color='red', linestyle='--')
        plt.title(f"{nome_modelo}: Real vs Previsto")
        plt.xlabel("Data")
        plt.ylabel("Inflação (%)")
        plt.legend()
        plt.grid(True)
        plt.show()
