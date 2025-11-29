# Previsão do IPCA com LSTM e Algoritmos Genéticos

![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-orange)
![DEAP](https://img.shields.io/badge/DEAP-1.3-lightgrey)
![Pandas](https://img.shields.io/badge/pandas-1.6-blue)
![NumPy](https://img.shields.io/badge/numpy-1.27-lightblue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green)
![Matplotlib](https://img.shields.io/badge/matplotlib-3.8-orange)
![Statsmodels](https://img.shields.io/badge/statsmodels-0.17-purple)
![Status](https://img.shields.io/badge/status-completo-brightgreen)


Este repositório contém o código e os dados para previsão do **Índice de Preços ao Consumidor Amplo (IPCA)** utilizando **Redes Neurais Recorrentes (LSTM)** combinadas com **Algoritmos Genéticos (GA)** para otimização de hiperparâmetros e seleção de variáveis. O projeto permite simular cenários futuros de inflação brasileira e avaliar a performance de modelos de séries temporais.

---

## 🔹 Artigo Completo

O artigo detalhado deste estudo, contendo metodologia, resultados e discussões, está disponível em PDF.

```
IPCA-LSTM-GA/
└─ docs/
   └─ Previsao_da_Inflacao_Brasileira_utilizando_Machine_Learning.pdf
```

Link para acessar ou baixar o PDF:

[📄 Baixar/Visualizar Artigo](Docs/Previsao_da_Inflacao_Brasileira_utilizando_Machine_Learning.pdf)

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
InflacaoBrasil_LSTM/
│
├─ Dados/
│   ├─ LSTM/
│   │   ├─ lstm_historico_previsao.png
│   │   └─ lstm_real_vs_previsto.png
│   │
│   ├─ RandomForest/
│   │   ├─ ipca_historico_previsao.png
│   │   └─ randomforest_real_vs_previsto.png
│   │
│   └─ RegressaoLinear/
│       ├─ ipca_previsao_reg_linear.png
│       └─ reglinear_real_vs_previsto.png
│
├─ docs/
│   └─ Previsao_da_Inflacao_Brasileira_utilizando_Machine_Learning.pdf
│
├─ __pycache__/
│   ├─ AvaliarModelo.cpython-311.pyc
│   ├─ PrevisaoFinal.cpython-311.pyc
│   └─ modelos.cpython-311.pyc
│
├─ data/
│   ├─ comparacao_modelo.csv
│   ├─ ipca_continuo.csv
│   └─ ipca_previsao_ml.csv
│
├─ .vscode/
│
├─ AvaliarModelo.py
├─ BaixarIPCA.py
├─ PrevisaoFinal.py
├─ README.md
├─ exploracao.py
├─ inflacao_base.py
├─ main.py
├─ modelos.py
└─ preprocessamento.py
```

---

## 🔹 Instalação e Uso

Clone o repositório, configure um ambiente virtual e instale todas as dependências para garantir execução isolada e sem conflitos. O pipeline modular inclui: pré-processamento de dados, treinamento da LSTM, otimização via Algoritmos Genéticos, avaliação de métricas e visualização de resultados. Os notebooks oferecem execução interativa e exploração detalhada das séries temporais e dos modelos.

```bash
git clone https://github.com/Scaglia05/InflacaoBrasil_LSTM.git
cd InflacaoBrasil_LSTM
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```


---

## 🔹 Resultados e Contribuições
O projeto gera previsões mensais do IPCA fora da amostra, métricas de desempenho confiáveis (RMSE, MAE, R²) e gráficos comparativos entre valores reais e previstos. Contribuições externas são bem-vindas: reporte problemas ou sugira melhorias via issues ou pull requests, ajudando a aprimorar a confiabilidade e replicabilidade do repositório.

---

## 🔹 Reprodutibilidade
Todos os scripts são versionados e configurados com parâmetros fixos (random_state=42), garantindo que execuções repetidas produzam resultados idênticos. Isso assegura consistência, validação confiável e possibilidade de comparações robustas entre diferentes ajustes de hiperparâmetros ou modelos.

---

## 🔹 Tecnologias

* Python 3.11
* TensorFlow 2.16
* Pandas, NumPy, Scikit-learn
* DEAP (Algoritmos Genéticos)
* Matplotlib, Seaborn
* GPU compatível para treinamento acelerado

---

<div align="center">
  <img src="Gifs/Machine.gif" width="320">
  <br>
  <em>Machine Learning </em>
</div>


