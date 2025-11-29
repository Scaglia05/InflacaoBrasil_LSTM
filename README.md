# Previsão do IPCA com LSTM e Algoritmos Genéticos

![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-orange)
![Status](https://img.shields.io/badge/status-completo-brightgreen)

Este repositório contém o código e os dados para previsão do **Índice de Preços ao Consumidor Amplo (IPCA)** utilizando **Redes Neurais Recorrentes (LSTM)** combinadas com **Algoritmos Genéticos (GA)** para otimização de hiperparâmetros e seleção de variáveis. O projeto permite simular cenários futuros de inflação brasileira e avaliar a performance de modelos de séries temporais.

---

## 🔹 Artigo Completo

O artigo detalhado deste estudo, contendo metodologia, resultados e discussões, está disponível em PDF. Para incluí-lo no repositório, coloque o arquivo em `docs/`:

```
IPCA-LSTM-GA/
└─ docs/
   └─ Previsao_da_Inflacao_Brasileira_utilizando_Machine_Learning.pdf
```

Link para acessar ou baixar o PDF:

[📄 Baixar/Visualizar Artigo](docs/Previsao_da_Inflacao_Brasileira_utilizando_Machine_Learning.pdf)

> Observação: o PDF também pode ser visualizado inline em navegadores que suportam embed de PDF, mas o link de download é a forma mais confiável.

---

## 🔹 Descrição do Projeto

O objetivo é construir um modelo preditivo capaz de:

* Capturar padrões temporais complexos do IPCA;
* Incorporar informações macroeconômicas relevantes;
* Otimizar hiperparâmetros e seleção de variáveis automaticamente;
* Avaliar a performance com métricas robustas (RMSE, MAE, R²).

A metodologia inclui:

* **LSTM**: captura tendências, sazonalidade e volatilidade do IPCA;
* **Algoritmos Genéticos**: otimizam hiperparâmetros e selecionam variáveis relevantes;
* **Pré-processamento de dados**: criação de lags, médias móveis, normalização e formatação temporal;
* **Avaliação e visualização**: cálculo de métricas e análise gráfica de resultados.

---

## 🔹 Estrutura do Repositório

```
IPCA-LSTM-GA/
│
├─ data/
│   ├─ ipca.csv
│   ├─ macro_vars.csv
│
├─ src/
│   ├─ preprocessing.py
│   ├─ model_lstm.py
│   ├─ genetic_optimizer.py
│   ├─ evaluation.py
│   ├─ visualization.py
│
├─ notebooks/
│   ├─ exploratory_analysis.ipynb
│   ├─ model_training.ipynb
│
├─ docs/
│   └─ Previsao_da_Inflacao_Brasileira_utilizando_Machine_Learning.pdf
│
├─ requirements.txt
└─ README.md
```

---

## 🔹 Instalação e Uso

Siga os passos descritos anteriormente para clonar, criar ambiente virtual, instalar dependências e executar os scripts de pré-processamento, treino, otimização, avaliação e visualização.

---

## 🔹 Resultados e Contribuições

Produz previsões mensais do IPCA fora da amostra, métricas (RMSE, MAE, R²) e gráficos de análise. Sugestões e melhorias podem ser enviadas via **issues** ou **pull requests**.

---

## 🔹 Reprodutibilidade

Scripts versionados, parâmetros fixos (`random_state=42`), garantindo resultados consistentes.

---

## 🔹 Tecnologias

* Python 3.11
* TensorFlow 2.16
* Pandas, NumPy, Scikit-learn
* DEAP (Algoritmos Genéticos)
* Matplotlib, Seaborn
* GPU compatível para treinamento acelerado
