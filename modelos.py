import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

LAGS = [1, 3, 6, 12]

def criar_lags(df, lags=LAGS):
    """Cria colunas de defasagem para séries temporais"""
    for lag in lags:
        df[f'lag_{lag}'] = df['valor'].shift(lag)
    return df

def treinar_rf(df, ate_ano='2020-12-01'):
    """Treina Random Forest"""
    df = criar_lags(df)
    df = df.dropna()
    
    train = df[:ate_ano]
    X_train = train[[f'lag_{lag}' for lag in LAGS]]
    y_train = train['valor']
    
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    return rf, X_train, y_train

def treinar_linear(df, ate_ano='2020-12-01'):
    """Treina Regressão Linear"""
    df = criar_lags(df)
    df = df.dropna()
    
    train = df[:ate_ano]
    X_train = train[[f'lag_{lag}' for lag in LAGS]]
    y_train = train['valor']
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    return lr, X_train, y_train

def treinar_lstm(df, ate_ano='2020-12-01', n_lags=12, epochs=100, batch_size=16):
    """
    Treina LSTM com dados até ate_ano.
    - n_lags: quantidade de meses anteriores para entrada
    """
    # Normalizar
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[['valor']]), columns=['valor'], index=df.index)
    
    # Criar sequências
    def criar_sequencias(data, n_lags):
        X, y = [], []
        for i in range(n_lags, len(data)):
            X.append(data[i-n_lags:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    train_data = df_scaled[:ate_ano].values
    X_train, y_train = criar_sequencias(train_data, n_lags)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # [samples, timesteps, features]
    
    # Modelo LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_lags, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    return model, X_train, y_train, scaler
