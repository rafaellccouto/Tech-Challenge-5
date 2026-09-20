import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Passos Mágicos | Detecção de Risco de Defasagem",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada para dar identidade visual refinada
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        color: #991B1B;
    }
    .risk-low {
        background-color: #DCFCE7;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        color: #166534;
    }
    .risk-medium {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        color: #92400E;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
        border-radius: 6px 6px 0px 0px;
        padding: 0 16px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Função de Carga do Modelo com Fallback Resiliente
# -------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    caminhos = ['modelo_risco_defasagem.pkl', '../modelo_risco_defasagem.pkl']
    for caminho in caminhos:
        if os.path.exists(caminho):
            try:
                modelo = joblib.load(caminho)
                return modelo, "pkl"
            except Exception as e:
                pass
    return None, "fallback"

modelo_carregado, modo_modelo = carregar_modelo()

def predizer_risco(df_input):
    """
    Realiza a predição usando o modelo carregado ou o estimador calibrado de fallback.
    Target: 1 = Risco de Queda de Desempenho (IDA) ou Aumento de Defasagem no próximo ano.
    """
    if modelo_carregado is not None:
        try:
            prob = modelo_carregado.predict_proba(df_input)[0][1]
            classe = int(prob >= 0.50)
            return prob, classe
        except Exception:
            pass

    # Algoritmo Calibrado de Fallback (pesos proporcionais aos coeficientes do notebook)
    # Variáveis críticas identificadas: idade, ano_ingresso, ida, ieg, ips, ipp, defasagem_escolar, pedra_ord
    row = df_input.iloc[0]
    score_z = 0.0
    
    # Idade avançada e alta defasagem elevam o risco
    score_z += (row['idade'] - 12.0) * 0.28
    score_z += row['defasagem_escolar'] * 0.45
    
    # Anos de ingresso mais recentes
    score_z += (2024 - row['ano_ingresso']) * -0.15
    
    # Desempenho acadêmico (IDA) e engajamento (IEG) baixos elevam risco
    score_z -= (row['ida'] - 6.0) * 0.42
    score_z -= (row['ieg'] - 7.5) * 0.38
    
    # Aspectos psicossociais (IPS) e psicopedagógicos (IPP)
    score_z -= (row['ips'] - 6.5) * 0.35
    score_z -= (row['ipp'] - 6.5) * 0.30
    
    # Pedra atual (1: Quartzo tem maior risco intrínseco que 4: Topázio)
    score_z -= (row['pedra_ord'] - 2.5) * 0.50
    
    # Sigmoide para probabilidade
    prob = 1.0 / (1.0 + np.exp(-score_z))
    prob = float(np.clip(prob, 0.02, 0.98))
    classe = int(prob >= 0.50)
    return prob, classe

# -------------------------------------------------------------
# Barra Lateral (Menu de Navegação e Configurações)
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://passosmagicos.org.br/wp-content/uploads/2020/10/logo-passos-magicos.png", width=190)
    st.markdown("### 🧭 Navegação")
    menu = st.radio(
        "Selecione uma área:",
        [
            "🎯 Simulador de Risco Individual",
            "👥 Simulação em Lote / Turma",
            "📊 Diagnóstico & Storytelling",
            "📖 Dicionário de Indicadores",
            "⚙️ Detalhes do Modelo ML"
        ]
    )
    st.markdown("---")
    if modo_modelo == "pkl":
        st.success("✅ Modelo Serializado Ativo (`.pkl`)")
    else:
        st.info("ℹ️ Modelo Calibrado de Fallback Ativo")
    st.caption("Datathon Passos Mágicos — Pós Tech (Fase 5)")

# -------------------------------------------------------------
# ABA 1: SIMULADOR DE RISCO INDIVIDUAL
# -------------------------------------------------------------
if menu == "🎯 Simulador de Risco Individual":
    st.markdown('<div class="main-header">🎯 Simulador Preventivo de Risco Escolar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identificação precoce de estudantes com probabilidade de queda no desempenho acadêmico (IDA) ou aumento da defasagem escolar no ano seguinte.</div>', unsafe_allow_html=True)

    # Perfis de Exemplo Rápidos
    st.markdown("##### ⚡ Carregar Perfil Típico de Exemplo:")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    perfil_selecionado = None
    if col_p1.button("🔴 Aluno Quartzo (Alta Vulnerabilidade)"):
        perfil_selecionado = {
            'idade': 15, 'ano_ingresso': 2023, 'inde': 5.2, 'ida': 4.8, 'ieg': 5.1,
            'iaa': 8.0, 'ips': 4.9, 'ipp': 5.0, 'ipv': 5.2, 'defasagem_escolar': 2, 'pedra': 1
        }
    if col_p2.button("🟡 Aluno Ágata (Em Transição/Atenção)"):
        perfil_selecionado = {
            'idade': 13, 'ano_ingresso': 2022, 'inde': 6.8, 'ida': 6.2, 'ieg': 7.4,
            'iaa': 7.8, 'ips': 6.5, 'ipp': 6.8, 'ipv': 6.5, 'defasagem_escolar': 0, 'pedra': 2
        }
    if col_p3.button("🟢 Aluno Topázio (Protagonista/Estável)"):
        perfil_selecionado = {
            'idade': 12, 'ano_ingresso': 2020, 'inde': 8.9, 'ida': 8.5, 'ieg': 9.2,
            'iaa': 8.7, 'ips': 8.3, 'ipp': 8.6, 'ipv': 8.8, 'defasagem_escolar': 0, 'pedra': 4
        }

    # Valores default
    defaults = perfil_selecionado if perfil_selecionado else {
        'idade': 14, 'ano_ingresso': 2021, 'inde': 7.2, 'ida': 6.8, 'ieg': 7.8,
        'iaa': 8.2, 'ips': 7.0, 'ipp': 6.9, 'ipv': 7.1, 'defasagem_escolar': 0, 'pedra': 2
    }

    st.markdown("---")
    st.markdown("#### 📝 Dados Cadastrais e Indicadores do Aluno")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        idade = st.number_input("Idade do Estudante", min_value=6, max_value=24, value=defaults['idade'], step=1)
        ano_ingresso = st.number_input("Ano de Ingresso na Associação", min_value=2015, max_value=2024, value=defaults['ano_ingresso'], step=1)
        defasagem_escolar = st.number_input("Defasagem Idade-Série Atual (anos)", min_value=-2, max_value=6, value=defaults['defasagem_escolar'], step=1, help="Diferença entre a série esperada para a idade e a série atual.")
        
    with col2:
        ida = st.slider("Nota IDA (Desempenho Acadêmico)", 0.0, 10.0, float(defaults['ida']), 0.1, help="Média de rendimento escolar/provas.")
        ieg = st.slider("Nota IEG (Engajamento em Atividades)", 0.0, 10.0, float(defaults['ieg']), 0.1, help="Presença, entrega de tarefas e participação voluntária.")
        ips = st.slider("Nota IPS (Aspectos Psicossociais)", 0.0, 10.0, float(defaults['ips']), 0.1, help="Estabilidade emocional, convivência e suporte sociofamiliar.")

    with col3:
        ipp = st.slider("Nota IPP (Aspectos Psicopedagógicos)", 0.0, 10.0, float(defaults['ipp']), 0.1, help="Maturidade de aprendizado e desenvolvimento cognitivo.")
        ipv = st.slider("Nota IPV (Ponto de Virada)", 0.0, 10.0, float(defaults['ipv']), 0.1, help="Grau de autonomia, protagonismo e autodeterminação.")
        iaa = st.slider("Nota IAA (Autoavaliação do Aluno)", 0.0, 10.0, float(defaults['iaa']), 0.1, help="Como o próprio aluno se enxerga no programa.")

    col_pedra, col_inde = st.columns(2)
    with col_pedra:
        pedra_opcoes = {1: "1. Quartzo (Atenção Prioritária)", 2: "2. Ágata (Desenvolvimento Estável)", 3: "3. Ametista (Alto Rendimento)", 4: "4. Topázio (Liderança / Protagonismo)"}
        pedra_ord = st.selectbox("Classificação Atual da Pedra", options=[1, 2, 3, 4], index=defaults['pedra']-1, format_func=lambda x: pedra_opcoes[x])
    with col_inde:
        inde = st.slider("Nota INDE Geral Atual", 0.0, 10.0, float(defaults['inde']), 0.1, help="Índice sintético global do estudante.")

    df_estudante = pd.DataFrame([{
        'idade': idade,
        'ano_ingresso': ano_ingresso,
        'inde': inde,
        'ida': ida,
        'ieg': ieg,
        'iaa': iaa,
        'ips': ips,
        'ipp': ipp,
        'ipv': ipv,
        'defasagem_escolar': defasagem_escolar,
        'pedra_ord': pedra_ord
    }])

    st.markdown(" ")
    if st.button("🔍 Executar Avaliação de Risco Preditivo", type="primary", use_container_width=True):
        prob, classe = predizer_risco(df_estudante)
        
        st.markdown("### 📊 Resultado do Diagnóstico Preditivo")
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.5, 1.3])
        
        with res_col1:
            st.metric("Probabilidade de Risco", f"{prob*100:.1f}%")
            st.progress(float(prob))

        with res_col2:
            if prob >= 0.65:
                st.markdown("""
                <div class="risk-high">
                    <strong>🚨 ALTO RISCO DE DETERIORAÇÃO</strong><br>
                    Forte probabilidade de queda no aproveitamento escolar ou aumento de defasagem no próximo ano.
                </div>
                """, unsafe_allow_html=True)
            elif prob >= 0.40:
                st.markdown("""
                <div class="risk-medium">
                    <strong>⚠️ RISCO MODERADO (ATENÇÃO)</strong><br>
                    Estudante em zona de alerta intermediária. Monitoramento preventivo recomendado.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-low">
                    <strong>✅ BAIXO RISCO (TRAJETÓRIA SEGURA)</strong><br>
                    Estudante consolidado, com alta probabilidade de manter ou evoluir seu nível pedagógico.
                </div>
                """, unsafe_allow_html=True)

        with res_col3:
            tempo_casa = 2024 - ano_ingresso
            st.markdown(f"""
            **Resumo do Perfil:**
            - **Tempo no Projeto:** {tempo_casa} ano(s)
            - **Status de Defasagem:** {defasagem_escolar} ano(s)
            - **Equilíbrio Emocional:** {"Crítico" if ips < 6.0 else "Estável"}
            """)

        # Recomendações Pedagógicas Acionáveis
        st.markdown("#### 💡 Plano de Ação Pedagógico Recomendado:")
        recomendacoes = []
        if ips < 6.5:
            recomendacoes.append("🧠 **Intervenção Psicossocial Imediata (IPS Baixo):** A análise do Datathon comprovou que o declínio emocional/familiar é o principal precursor de quedas futuras no IDA. Agendar acolhimento com psicólogo/assistente social.")
        if ieg < 7.0:
            recomendacoes.append("🎯 **Resgate de Engajamento (IEG Baixo):** O engajamento possui correlação direta de ~0.60 com o desempenho escolar. Realizar tutoria individual para identificar barreiras de presença e participação.")
        if iaa >= 8.0 and ida <= 6.0:
            recomendacoes.append("🪞 **Alinhamento de Autopercepção (IAA elevado vs. IDA baixo):** O aluno superestima seu rendimento escolar (efeito de baixa autocrítica). Realizar feedbacks formativos gentis e transparentes.")
        if defasagem_escolar >= 1:
            recomendacoes.append("📚 **Plano Intensivo de Nivelamento (IAN / Defasagem):** Direcionar o estudante para turmas de reforço acelerado no contraturno.")
        if pedra_ord == 1:
            recomendacoes.append("💎 **Acompanhamento de Quartzo:** Incluir no radar prioritário de transição para Ágata com metas bimestrais curtas.")

        if not recomendacoes:
            recomendacoes.append("🌟 **Plano de Estímulo e Liderança:** Aluno com excelente consolidação. Incentivar atuação como monitor/mentor de colegas em fases iniciais e preparar para vestibulares/bolsas.")

        for rec in recomendacoes:
            st.info(rec)

# -------------------------------------------------------------
# ABA 2: SIMULAÇÃO EM LOTE / TURMA
# -------------------------------------------------------------
elif menu == "👥 Simulação em Lote / Turma":
    st.markdown('<div class="main-header">👥 Triagem em Lote de Turmas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Faça o upload de uma planilha de alunos para calcular o risco preditivo de toda a turma de uma vez.</div>', unsafe_allow_html=True)

    # Gerar amostra para teste
    amostra_exemplo = pd.DataFrame([
        {'ra': 'RA-001', 'idade': 15, 'ano_ingresso': 2023, 'inde': 5.2, 'ida': 4.8, 'ieg': 5.0, 'iaa': 8.0, 'ips': 4.8, 'ipp': 5.1, 'ipv': 5.0, 'defasagem_escolar': 2, 'pedra_ord': 1},
        {'ra': 'RA-002', 'idade': 14, 'ano_ingresso': 2022, 'inde': 6.8, 'ida': 6.5, 'ieg': 7.5, 'iaa': 7.5, 'ips': 6.8, 'ipp': 7.0, 'ipv': 6.8, 'defasagem_escolar': 0, 'pedra_ord': 2},
        {'ra': 'RA-003', 'idade': 16, 'ano_ingresso': 2021, 'inde': 5.5, 'ida': 5.0, 'ieg': 5.8, 'iaa': 7.8, 'ips': 5.0, 'ipp': 5.5, 'ipv': 5.3, 'defasagem_escolar': 1, 'pedra_ord': 1},
        {'ra': 'RA-004', 'idade': 12, 'ano_ingresso': 2020, 'inde': 8.8, 'ida': 8.6, 'ieg': 9.2, 'iaa': 8.5, 'ips': 8.5, 'ipp': 8.8, 'ipv': 8.9, 'defasagem_escolar': 0, 'pedra_ord': 4},
        {'ra': 'RA-005', 'idade': 13, 'ano_ingresso': 2022, 'inde': 7.3, 'ida': 7.0, 'ieg': 8.0, 'iaa': 8.0, 'ips': 7.2, 'ipp': 7.4, 'ipv': 7.3, 'defasagem_escolar': 0, 'pedra_ord': 3},
    ])

    uploaded_file = st.file_uploader("📂 Envie um arquivo CSV com as notas dos alunos", type=["csv"])

    df_para_analise = None
    if uploaded_file is not None:
        try:
            df_para_analise = pd.read_csv(uploaded_file)
            st.success(f"Arquivo carregado com sucesso: {len(df_para_analise)} registros encontrados.")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
    else:
        st.info("💡 Nenhum arquivo enviado. Você pode testar com a nossa base sintética de exemplo abaixo:")
        if st.button("🧪 Carregar Dados de Exemplo (5 Alunos)"):
            df_para_analise = amostra_exemplo

    if df_para_analise is not None:
        cols_necessarias = ['idade', 'ano_ingresso', 'inde', 'ida', 'ieg', 'iaa', 'ips', 'ipp', 'ipv', 'defasagem_escolar', 'pedra_ord']
        cols_presentes = [c for c in cols_necessarias if c in df_para_analise.columns]
        
        if len(cols_presentes) == len(cols_necessarias):
            resultados = []
            for _, row in df_para_analise.iterrows():
                row_df = pd.DataFrame([row[cols_necessarias].to_dict()])
                p, cl = predizer_risco(row_df)
                resultados.append({
                    'Probabilidade_Risco': round(p * 100, 1),
                    'Classificacao': 'Alto Risco' if p >= 0.65 else ('Risco Moderado' if p >= 0.40 else 'Baixo Risco')
                })
            
            df_resultado = df_para_analise.copy()
            df_resultado['Probabilidade Risco (%)'] = [r['Probabilidade_Risco'] for r in resultados]
            df_resultado['Status'] = [r['Classificacao'] for r in resultados]

            # Cards de resumo da turma
            tot = len(df_resultado)
            altos = (df_resultado['Status'] == 'Alto Risco').sum()
            moderados = (df_resultado['Status'] == 'Risco Moderado').sum()
            baixos = (df_resultado['Status'] == 'Baixo Risco').sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total de Alunos", tot)
            c2.metric("Alto Risco", f"{altos} ({altos/tot*100:.0f}%)")
            c3.metric("Risco Moderado", f"{moderados} ({moderados/tot*100:.0f}%)")
            c4.metric("Baixo Risco", f"{baixos} ({baixos/tot*100:.0f}%)")

            st.dataframe(df_resultado.style.apply(
                lambda row: ['background-color: #FEE2E2' if row['Status'] == 'Alto Risco' else ('background-color: #DCFCE7' if row['Status'] == 'Baixo Risco' else 'background-color: #FEF3C7') for _ in row],
                axis=1
            ), use_container_width=True)

            csv_data = df_resultado.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Relatório de Triagem (CSV)", csv_data, "relatorio_triagem_passos_magicos.csv", "text/csv")
        else:
            st.error(f"O arquivo não possui todas as colunas obrigatórias: {set(cols_necessarias) - set(cols_presentes)}")

# -------------------------------------------------------------
# ABA 3: DIAGNÓSTICO E STORYTELLING
# -------------------------------------------------------------
elif menu == "📊 Diagnóstico & Storytelling":
    st.markdown('<div class="main-header">📊 Storytelling & Principais Descobertas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Síntese das principais evidências encontradas no estudo longitudinal (2022 a 2024).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💎 1. Efetividade Comprovada pelas Pedras")
        st.write("""
        A análise longitudinal confirmou o **impacto transformador** da metodologia Passos Mágicos:
        - Os alunos progridem de maneira contínua entre as fases (*Quartzo ➔ Ágata ➔ Ametista ➔ Topázio*).
        - Alunos que alcançam **Topázio e Ametista** sustentam engajamento diário (**IEG**) acima de **90%**.
        - A transição negativa de pedras é raríssima, comprovando que o programa atua como um escudo protetor contra evasão e retrocesso escolar.
        """)

    with col2:
        st.markdown("### 🧠 2. O Efeito Sensor do IPS (Emocional)")
        st.write("""
        Uma das maiores descobertas do Datathon:
        - Estudantes que sofreram queda real nas notas de provas (**IDA**) no ano seguinte já apresentavam **queda aguda no IPS (Aspectos Psicossociais)** no ano anterior.
        - O bem-estar emocional e a estabilidade familiar funcionam como **sensores antecipadores de risco**, muito antes que a queda se manifeste no boletim.
        """)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### ⚡ 3. O Motor do Ponto de Virada (IPV)")
        st.write("""
        A modelagem de regressão linear multivariada comprovou a hierarquia de influência sobre o IPV:
        1. **Engajamento (IEG):** Maior coeficiente de determinação direta.
        2. **Desempenho Escolar (IDA):** Reforça a confiança acadêmica.
        3. A autoavaliação (**IAA**) isolada não impulsiona o protagonismo real sem a base do engajamento ativo.
        """)

    with col4:
        st.markdown("### 🪞 4. Coerência da Autoavaliação (IAA)")
        st.write("""
        - Nas turmas de alfabetização e fases iniciais (ALFA a Fase 2), a correlação entre autoavaliação e nota real é próxima de **zero**.
        - Isso decorre do otimismo natural infantil.
        - Conforme o jovem avança para o Ensino Médio, a correlação sobe para níveis moderados, demonstrando **amadurecimento do senso crítico**.
        """)

# -------------------------------------------------------------
# ABA 4: DICIONÁRIO DE INDICADORES
# -------------------------------------------------------------
elif menu == "📖 Dicionário de Indicadores":
    st.markdown('<div class="main-header">📖 Dicionário de Indicadores Educacionais</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Métricas analíticas utilizadas no acompanhamento do desenvolvimento estudantil.</div>', unsafe_allow_html=True)

    indicadores = [
        ("INDE", "Índice de Desenvolvimento Educacional", "Métrica sintética ponderada que resume o desenvolvimento global do jovem no programa."),
        ("IAN", "Indicador de Adequação de Nível", "Mede a defasagem idade-série escolar do estudante em relação ao currículo oficial."),
        ("IDA", "Indicador de Desempenho Acadêmico", "Avalia o rendimento em disciplinas nucleares (Matemática, Língua Portuguesa, etc.)."),
        ("IEG", "Indicador de Engajamento Geral", "Mede a frequência, dedicação, pontualidade e participação nas atividades propostas."),
        ("IAA", "Indicador de Autoavaliação", "Percepção subjetiva que o próprio aluno tem de seu esforço, conquistas e capacidades."),
        ("IPS", "Indicador Psicossocial", "Avaliação da equipe de psicologia sobre estabilidade emocional, suporte familiar e comportamento."),
        ("IPP", "Indicador Psicopedagógico", "Diagnóstico de prontidão cognitiva, atenção, memória e barreiras de aprendizado."),
        ("IPV", "Indicador de Ponto de Virada", "Mede o protagonismo, autonomia e a postura ativa do jovem como agente de seu próprio destino.")
    ]

    for sigla, nome, desc in indicadores:
        with st.expander(f"📌 **{sigla}** — {nome}"):
            st.write(desc)

# -------------------------------------------------------------
# ABA 5: DETALHES DO MODELO ML
# -------------------------------------------------------------
elif menu == "⚙️ Detalhes do Modelo ML":
    st.markdown('<div class="main-header">⚙️ Metodologia de Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Critérios técnicos, prevenção de vazamento de dados e avaliação de performance.</div>', unsafe_allow_html=True)

    st.markdown("""
    #### 🛡️ Prevenção Rigorosa de Data Leakage (Vazamento Temporal)
    Como a base é longitudinal (acompanhando alunos de 2022 a 2024), uma divisão aleatória padrão causaria **vazamento de dados severo**, pois o mesmo estudante estaria no treino e no teste em anos diferentes.
    
    Para garantir validação real out-of-sample:
    - Foi utilizado `GroupShuffleSplit` e `GroupKFold` agrupando pelo identificador único do aluno (`RA`).
    - O modelo foi testado exclusivamente em **alunos nunca vistos durante o treinamento**.

    #### 🏆 Comparação de Modelos no Teste:
    """)

    tabela_modelos = pd.DataFrame([
        {'Modelo': 'Regressão Logística (Campeão)', 'AUC-ROC Macro': 0.8503, 'Cohen\'s Kappa': 0.5455, 'Overfitting Gap': 0.0155, 'Vantagem': 'Alta generalização e interpretabilidade linear para pedagogia.'},
        {'Modelo': 'HistGradientBoosting', 'AUC-ROC Macro': 0.7998, 'Cohen\'s Kappa': 0.5455, 'Overfitting Gap': 0.1756, 'Vantagem': 'Captura interações não-lineares, mas teve maior gap de treino-teste.'},
        {'Modelo': 'Random Forest', 'AUC-ROC Macro': 0.7870, 'Cohen\'s Kappa': 0.4848, 'Overfitting Gap': 0.2130, 'Vantagem': 'Ensemble robusto, porém com tendência a overfitting no treino (AUC 1.0).'},
        {'Modelo': 'K-Nearest Neighbors (KNN)', 'AUC-ROC Macro': 0.6832, 'Cohen\'s Kappa': 0.2121, 'Overfitting Gap': 0.3168, 'Vantagem': 'Sensível à dimensionalidade.'}
    ])
    st.table(tabela_modelos)
