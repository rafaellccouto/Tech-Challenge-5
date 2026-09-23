# Tech-Challenge-5

Projeto de análise e modelagem preditiva para identificação de risco de defasagem escolar no contexto do Datathon Passos Mágicos.

Autores:
- Rafael Couto
- Alex Oliveira
- Ronaldo Rodrigues

## 📌 Visão geral

Este repositório reúne o pipeline completo de análise, consolidação de dados, engenharia de features e modelagem preditiva para identificar risco de defasagem escolar e queda de desempenho acadêmico.

O objetivo principal foi transformar indicadores pedagógicos e comportamentais em uma solução operacional capaz de sinalizar estudantes em maior risco de deterioração no ano seguinte.

A parte principal da análise está centralizada no notebook:

- `pipeline_datathon_passos_magicos_refactored.ipynb` — pipeline completo com limpeza, engenharia de features, comparação de modelos, validação e export do artefato final.

A aplicação interativa fica separada do fluxo analítico:

- `streamlit_app.py` — app de previsão para uso operacional.

## 🧾 Dados utilizados

A base original foi distribuída em três abas do arquivo Excel:

- `PEDE2022`
- `PEDE2023`
- `PEDE2024`

Volumes observados:

- 2022: 860 registros
- 2023: 1.014 registros
- 2024: 1.156 registros

A consolidação foi feita por aluno (`ra`) para preservar a estrutura longitudinal do problema. Os indicadores principais incluíram:

- `inde` — índice de desempenho geral
- `ian` — adequação de nível / defasagem escolar
- `ida` — desempenho acadêmico
- `ieg` — engajamento
- `ipp`, `ips`, `ipv` — indicadores psicopedagógico, psicossocial e de ponto de virada
- `defasagem_escolar` — diferença entre a série esperada e a série atual
- `delta_inde` — variação do desempenho ao longo do tempo
- `taxa_engajamento` — indicador de envolvimento do aluno

A variável alvo foi construída para capturar risco futuro por aluno, considerando aumento de defasagem e/ou queda de desempenho no ano seguinte.

## 🧭 O que foi feito

- Carregamento e inspeção das três bases históricas.
- Padronização de nomes de colunas e conversão de tipos numéricos.
- Consolidação longitudinal por aluno para preservar a estrutura temporal.
- Engenharia de features com indicadores de desempenho, engajamento e risco.
- Definição do alvo de risco de defasagem escolar.
- Análise exploratória para responder perguntas de negócio sobre adequação de nível, desempenho e engajamento.
- Treinamento e comparação de modelos de classificação.
- Validação com cuidado para reduzir risco de leakage e overfitting.
- Explicabilidade e diagnóstico dos fatores de maior impacto.
- Export do modelo treinado em artefato serializado para uso posterior.

## 🔎 Principais descobertas

Os principais achados da análise apontam para um padrão consistente: o risco de defasagem escolar está associado principalmente à queda no desempenho acadêmico e ao aumento da defasagem ao longo do tempo.

Principais conclusões:

- o risco aumenta quando há queda no IDA e/ou elevação da defasagem escolar no período seguinte;
- a evolução do INDE e o comportamento longitudinal do aluno ajudaram a distinguir risco real de variação pontual;
- as variáveis de desempenho e engajamento tiveram maior contribuição para a previsão;
- indicadores como `ian`, `inde`, `ida`, `ieg` e `defasagem_escolar` foram os mais relevantes para sinalizar deterioração precoce;
- a validação por grupo e a estrutura do pipeline reduziram risco de overfitting e vazamento de informação.

A análise também mostrou concordância entre os indicadores de defasagem acadêmica e as avaliações psicopedagógicas, reforçando que a defasagem é um fenômeno de intervenção pedagógica e apoio, não apenas de gestão administrativa.

## 📊 Resultado executivo

### Perfil de defasagem

A distribuição observada conforme o IAN foi:

- Defasagem moderada: 793 alunos (57,93%)
- Adequado (sem defasagem): 560 alunos (40,91%)
- Defasagem severa: 16 alunos (1,17%)

Esse resultado reforça a relevância de ações preventivas e de monitoramento contínuo, principalmente no segmento de defasagem moderada, que representa a maior parte dos casos.

### Modelo final selecionado

O modelo final adotado foi a Regressão Logística, selecionada após comparação com KNN, Random Forest e Gradient Boosting Ensemble.

Justificativa da escolha:

- melhor desempenho geral em AUC ROC;
- boa capacidade de discriminação entre alunos em risco e não risco;
- maior estabilidade em validação e menor risco de overfitting em comparação com modelos mais flexíveis.

### Performance do modelo final

- AUC ROC: 0,8503
- Acurácia: 0,77

### Relatório de classificação

- Sem Risco: precision 0,80; recall 0,73; f1-score 0,76
- Em Risco (Queda/Defasagem Futura): precision 0,75; recall 0,82; f1-score 0,78

Esses indicadores demonstram que o modelo possui boa aderência ao problema e potencial de uso em monitoramento acadêmico e apoio à priorização de intervenções.

## 🏗️ Estrutura do projeto

```text
Tech-Challenge-5/
├── Dados_Base/
│   └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
├── pipeline_datathon_passos_magicos_refactored.ipynb
├── streamlit_app.py
├── modelo_risco_defasagem.pkl
├── README.md
├── requirements.txt
├── .gitignore
└── arquivos auxiliares de análise e validação
```

## ✅ Resultado principal

O notebook final produz um modelo de classificação pronto para prever risco de defasagem escolar e queda de desempenho com base em indicadores estruturados da base pedagógica.

A solução foi organizada em um pipeline narrativo e reproduzível, com foco em clareza, rastreabilidade e uso operacional. O modelo final selecionado foi a Regressão Logística, com AUC ROC de 0,8503 e desempenho adequado para suporte de decisão em contexto escolar.

## ▶️ Como executar

1. Abra o projeto em um ambiente Python com as dependências instaladas.
2. Ative o ambiente virtual do projeto.
3. Instale as bibliotecas listadas em `requirements.txt` caso ainda não estejam disponíveis.
4. Abra o notebook `pipeline_datathon_passos_magicos_refactored.ipynb`.
5. Execute as células em ordem para reproduzir a análise, o treinamento e a exportação do modelo.
6. Para uso operacional, execute o aplicativo Streamlit com o comando:

```bash
streamlit run streamlit_app.py
```

## 🧪 Dependências principais

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- openpyxl
- streamlit

## 📝 Observações

- O notebook principal é o artefato de referência do projeto.
- O app em Streamlit foi mantido separado do fluxo analítico, como solicitado.
- O projeto foi organizado para reduzir ruído de arquivos auxiliares e manter o foco na solução final.
- O estudo enfatiza uso preventivo, priorização de intervenções e suporte à tomada de decisão pedagógica.

## 🎯 Objetivo do desafio

A proposta deste trabalho foi transformar a base de dados em sinais de risco pedagógico e converter esses sinais em um modelo útil para apoio à decisão, identificação precoce de necessidades, priorização de intervenções e monitoramento de estudantes em maior vulnerabilidade acadêmica.
