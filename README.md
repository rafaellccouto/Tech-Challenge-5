# Tech-Challenge-5

Projeto de análise e modelagem preditiva para identificação de risco de defasagem escolar no contexto do Datathon Passos Mágicos.

## 📌 Visão geral

Este repositório reúne o pipeline completo de análise, desde a consolidação dos dados até a geração de um modelo preditivo e a preparação da solução para uso em aplicações.

A parte principal da solução está centralizada no notebook:

- `pipeline_datathon_passos_magicos.ipynb` — notebook único com narrativa, análise exploratória, engenharia de features, validação e export do modelo.

A aplicação interativa fica separada do fluxo analítico, conforme solicitado:

- `streamlit_app.py` — app de previsão para uso operacional.

## 🧭 O que foi feito

- Carregamento e inspeção das bases dos anos 2022, 2023 e 2024.
- Padronização de nomes de colunas e conversão de valores numéricos.
- Consolidação longitudinal por aluno, preservando a estrutura temporal do problema.
- Engenharia de features com indicadores de desempenho, engajamento e defasagem.
- Construção da variável alvo de risco de defasagem escolar.
- Análise exploratória para responder perguntas de negócio sobre desempenho e risco.
- Treinamento e comparação de modelos de classificação.
- Validação com cuidado para reduzir risco de leakage e overfitting.
- Explicabilidade das variáveis mais importantes para a previsão.
- Export do modelo treinado em artefato serializado para uso posterior.

## 🔎 Principais descobertas

Ao longo da análise, o projeto identificou que a defasagem escolar está fortemente associada a indicadores de desempenho e engajamento, especialmente em métricas que capturam evolução ao longo do tempo.

Entre os principais achados:

- a variável de defasagem e indicadores relacionados tiveram forte peso na previsão;
- a evolução do INDE e o comportamento longitudinal do aluno ajudaram a distinguir risco real de simples variação pontual;
- o engajamento e a performance acadêmica se mostraram relevantes para sinalizar deterioração precoce;
- a validação por grupo e a estrutura do pipeline ajudaram a reduzir o risco de overfitting e de vazamento de informação.

## 🏗️ Estrutura do projeto

```text
Tech-Challenge-5/
├── Dados_Base/
│   └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
├── pipeline_datathon_passos_magicos.ipynb
├── streamlit_app.py
├── modelo_risco_defasagem.pkl
├── README.md
├── .gitignore
├── requirements.txt
└── outros arquivos auxiliares de análise e validação
```

## ✅ Resultado principal

O notebook final produz um modelo de classificação pronto para prever risco de defasagem escolar com base em indicadores estruturados da base pedagógica.

A solução foi organizada em um pipeline narrativo e documentado, mantendo o foco em reproduzibilidade, clareza e uso operacional.

## ▶️ Como executar

1. Abra o projeto em um ambiente Python com as dependências instaladas.
2. Ative o ambiente virtual do projeto.
3. Instale as bibliotecas listadas em `requirements.txt` caso ainda não estejam disponíveis.
4. Abra o notebook `pipeline_datathon_passos_magicos.ipynb`.
5. Execute as células em ordem para reproduzir a análise e o treinamento do modelo.

## 🧪 Dependências principais

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- openpyxl

## 📝 Observações

- O notebook principal é o artefato de referência do projeto.
- O app em Streamlit foi mantido separado do fluxo analítico, como solicitado.
- O projeto foi organizado para reduzir ruído de arquivos auxiliares e manter o foco na solução final.

## 🎯 Objetivo do desafio

A proposta deste trabalho foi transformar a base de dados em um conjunto de sinais de risco pedagógico e transformar esses sinais em um modelo útil para apoio à tomada de decisão, identificação precoce de necessidades e priorização de intervenções.
