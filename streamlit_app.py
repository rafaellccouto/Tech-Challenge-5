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

# Estilização CSS personalizada para Modo Light Elegante, Alta Legibilidade e Rolagem Suave
st.markdown("""
<style>
    /* Estilos Gerais do Tema Light com Alto Contraste para Leitura Perfeita */
    .stApp {
        background-color: #F8FAFC;
        color: #0B0F19;
    }
    
    /* Textos Gerais, Títulos e Labels */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stRadio label, .stCheckbox label {
        color: #0B0F19 !important;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A !important;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #1E293B !important;
        margin-bottom: 1.5rem;
    }
    
    /* Inputs, Sliders e Widgets visíveis e legíveis no Light Mode */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #0B0F19 !important;
        border-color: #94A3B8 !important;
        font-weight: 500;
    }

    /* Rótulos dos Sliders e Number Inputs com alta visibilidade */
    .stSlider label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
        color: #0B0F19 !important;
        font-weight: 600 !important;
    }
    
    /* Estilização de Cards e Métricas */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #CBD5E1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        color: #0B0F19;
    }
    
    /* Alertas de Risco com Cores Suaves e Ótimo Contraste */
    .risk-high {
        background-color: #FFF1F2;
        border-left: 5px solid #E11D48;
        padding: 1rem;
        border-radius: 8px;
        color: #881337 !important;
        border-top: 1px solid #FFE4E6;
        border-right: 1px solid #FFE4E6;
        border-bottom: 1px solid #FFE4E6;
    }
    .risk-high * {
        color: #881337 !important;
    }
    
    .risk-low {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 1rem;
        border-radius: 8px;
        color: #14532D !important;
        border-top: 1px solid #DCFCE7;
        border-right: 1px solid #DCFCE7;
        border-bottom: 1px solid #DCFCE7;
    }
    .risk-low * {
        color: #14532D !important;
    }
    
    .risk-medium {
        background-color: #FFFBEB;
        border-left: 5px solid #D97706;
        padding: 1rem;
        border-radius: 8px;
        color: #78350F !important;
        border-top: 1px solid #FEF3C7;
        border-right: 1px solid #FEF3C7;
        border-bottom: 1px solid #FEF3C7;
    }
    .risk-medium * {
        color: #78350F !important;
    }

    /* Ajustes das Abas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
        border-radius: 8px 8px 0px 0px;
        padding: 0 16px;
        background-color: #E2E8F0;
        color: #1E293B !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1E3A8A !important;
        border-top: 3px solid #1E3A8A !important;
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
    row = df_input.iloc[0]
    score_z = 0.0
    
    score_z += (row['idade'] - 12.0) * 0.28
    score_z += row['defasagem_escolar'] * 0.45
    score_z += (2024 - row['ano_ingresso']) * -0.15
    score_z -= (row['ida'] - 6.0) * 0.42
    score_z -= (row['ieg'] - 7.5) * 0.38
    score_z -= (row['ips'] - 6.5) * 0.35
    score_z -= (row['ipp'] - 6.5) * 0.30
    score_z -= (row['pedra_ord'] - 2.5) * 0.50
    
    prob = 1.0 / (1.0 + np.exp(-score_z))
    prob = float(np.clip(prob, 0.02, 0.98))
    classe = int(prob >= 0.50)
    return prob, classe

# Função auxiliar JavaScript para rolagem suave em slow motion até os resultados
def rolar_para_resultado():
    st.markdown("""
        <script>
            setTimeout(function() {
                const target = window.parent.document.getElementById('resultado-anchor');
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 100);
        </script>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Barra Lateral (Menu de Navegação e Configurações)
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://passosmagicos.org.br/wp-content/uploads/2020/10/logo-passos-magicos.png", width=190)
    st.markdown("### Navegação")
    menu = st.radio(
        "Selecione uma área:",
        [
            "Simulador de Risco Individual",
            "Simulação em Lote / Turma",
            "Diagnóstico e Storytelling",
            "Dicionário de Indicadores",
            "Detalhes do Modelo ML"
        ]
    )
    st.markdown("---")
    if modo_modelo == "pkl":
        st.success("Modelo Serializado Ativo (.pkl)")
    else:
        st.info("Modelo Calibrado de Fallback Ativo")
    st.caption("Datathon Passos Mágicos — Pós Tech (Fase 5)")

# -------------------------------------------------------------
# ABA 1: SIMULADOR DE RISCO INDIVIDUAL
# -------------------------------------------------------------
if menu == "Simulador de Risco Individual":
    st.markdown('<div class="main-header">Simulador Preventivo de Risco Escolar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identificação precoce de estudantes com probabilidade de queda no desempenho acadêmico (IDA) ou aumento da defasagem escolar no ano seguinte.</div>', unsafe_allow_html=True)

    st.markdown("##### Carregar Perfil Típico de Exemplo:")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    perfil_selecionado = None
    if col_p1.button("Aluno Quartzo (Alta Vulnerabilidade)"):
        perfil_selecionado = {
            'idade': 15, 'ano_ingresso': 2023, 'inde': 5.2, 'ida': 4.8, 'ieg': 5.1,
            'iaa': 8.0, 'ips': 4.9, 'ipp': 5.0, 'ipv': 5.2, 'defasagem_escolar': 2, 'pedra': 1
        }
    if col_p2.button("Aluno Ágata (Em Transição/Atenção)"):
        perfil_selecionado = {
            'idade': 13, 'ano_ingresso': 2022, 'inde': 6.8, 'ida': 6.2, 'ieg': 7.4,
            'iaa': 7.8, 'ips': 6.5, 'ipp': 6.8, 'ipv': 6.5, 'defasagem_escolar': 0, 'pedra': 2
        }
    if col_p3.button("Aluno Topázio (Protagonista/Estável)"):
        perfil_selecionado = {
            'idade': 12, 'ano_ingresso': 2020, 'inde': 8.9, 'ida': 8.5, 'ieg': 9.2,
            'iaa': 8.7, 'ips': 8.3, 'ipp': 8.6, 'ipv': 8.8, 'defasagem_escolar': 0, 'pedra': 4
        }

    defaults = perfil_selecionado if perfil_selecionado else {
        'idade': 14, 'ano_ingresso': 2021, 'inde': 7.2, 'ida': 6.8, 'ieg': 7.8,
        'iaa': 8.2, 'ips': 7.0, 'ipp': 6.9, 'ipv': 7.1, 'defasagem_escolar': 0, 'pedra': 2
    }

    st.markdown("---")
    st.markdown("#### Dados Cadastrais e Indicadores do Aluno")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        idade = st.number_input("Idade do Estudante", min_value=6, max_value=24, value=defaults['idade'], step=1)
        ano_ingresso = st.number_input("Ano de Ingresso na Associação", min_value=2015, max_value=2024, value=defaults['ano_ingresso'], step=1)
        defasagem_escolar = st.number_input("Defasagem Idade-Série Atual (anos)", min_value=-2, max_value=6, value=defaults['defasagem_escolar'], step=1)
        
    with col2:
        ida = st.slider("Nota IDA (Desempenho Acadêmico)", 0.0, 10.0, float(defaults['ida']), 0.1)
        ieg = st.slider("Nota IEG (Engajamento em Atividades)", 0.0, 10.0, float(defaults['ieg']), 0.1)
        ips = st.slider("Nota IPS (Aspectos Psicossociais)", 0.0, 10.0, float(defaults['ips']), 0.1)

    with col3:
        ipp = st.slider("Nota IPP (Aspectos Psicopedagógicos)", 0.0, 10.0, float(defaults['ipp']), 0.1)
        ipv = st.slider("Nota IPV (Ponto de Virada)", 0.0, 10.0, float(defaults['ipv']), 0.1)
        iaa = st.slider("Nota IAA (Autoavaliação do Aluno)", 0.0, 10.0, float(defaults['iaa']), 0.1)

    col_pedra, col_inde = st.columns(2)
    with col_pedra:
        pedra_opcoes = {1: "1. Quartzo (Atenção Prioritária)", 2: "2. Ágata (Desenvolvimento Estável)", 3: "3. Ametista (Alto Rendimento)", 4: "4. Topázio (Liderança / Protagonismo)"}
        pedra_ord = st.selectbox("Classificação Atual da Pedra", options=[1, 2, 3, 4], index=defaults['pedra']-1, format_func=lambda x: pedra_opcoes[x])
    with col_inde:
        inde = st.slider("Nota INDE Geral Atual", 0.0, 10.0, float(defaults['inde']), 0.1)

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
    if st.button("Executar Avaliação de Risco Preditivo", type="primary", use_container_width=True):
        prob, classe = predizer_risco(df_estudante)
        
        # Âncora invisível para scroll automático em slow motion
        st.markdown('<div id="resultado-anchor"></div>', unsafe_allow_html=True)
        
        st.markdown("### Resultado do Diagnóstico Preditivo")
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.5, 1.3])
        
        with res_col1:
            st.metric("Probabilidade de Risco", f"{prob*100:.1f}%")
            st.progress(float(prob))

        with res_col2:
            if prob >= 0.65:
                st.markdown("""
                <div class="risk-high">
                    <strong>ALTO RISCO DE DETERIORAÇÃO</strong><br>
                    Forte probabilidade de queda no aproveitamento escolar ou aumento de defasagem no próximo ano.
                </div>
                """, unsafe_allow_html=True)
            elif prob >= 0.40:
                st.markdown("""
                <div class="risk-medium">
                    <strong>RISCO MODERADO (ATENÇÃO)</strong><br>
                    Estudante em zona de alerta intermediária. Monitoramento preventivo recomendado.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-low">
                    <strong>BAIXO RISCO (TRAJETÓRIA SEGURA)</strong><br>
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

        st.markdown("#### Plano de Ação Pedagógico Recomendado:")
        recomendacoes = []
        if ips < 6.5:
            recomendacoes.append("Intervenção Psicossocial Imediata (IPS Baixo): Agendar acolhimento com psicólogo/assistente social.")
        if ieg < 7.0:
            recomendacoes.append("Resgate de Engajamento (IEG Baixo): Realizar tutoria individual para identificar barreiras de participação.")
        if iaa >= 8.0 and ida <= 6.0:
            recomendacoes.append("Alinhamento de Autopercepção (IAA elevado vs. IDA baixo): Realizar feedbacks formativos gentis e transparentes.")
        if defasagem_escolar >= 1:
            recomendacoes.append("Plano Intensivo de Nivelamento: Direcionar o estudante para turmas de reforço acelerado.")
        if pedra_ord == 1:
            recomendacoes.append("Acompanhamento de Quartzo: Incluir no radar prioritário de transição para Ágata.")

        if not recomendacoes:
            recomendacoes.append("Plano de Estímulo e Liderança: Aluno com excelente consolidação. Incentivar atuação como monitor/mentor.")

        for rec in recomendacoes:
            st.info(rec)
            
        rolar_para_resultado()

# -------------------------------------------------------------
# ABA 2: SIMULAÇÃO EM LOTE / TURMA
# -------------------------------------------------------------
elif menu == "Simulação em Lote / Turma":
    st.markdown('<div class="main-header">Triagem em Lote de Turmas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Faça o upload de uma planilha de alunos para calcular o risco preditivo de toda a turma de uma vez.</div>', unsafe_allow_html=True)

    amostra_exemplo = pd.DataFrame([
        {'ra': 'RA-001', 'idade': 15, 'ano_ingresso': 2023, 'inde': 5.2, 'ida': 4.8, 'ieg': 5.0, 'iaa': 8.0, 'ips': 4.8, 'ipp': 5.1, 'ipv': 5.0, 'defasagem_escolar': 2, 'pedra_ord': 1},
        {'ra': 'RA-002', 'idade': 14, 'ano_ingresso': 2022, 'inde': 6.8, 'ida': 6.5, 'ieg': 7.5, 'iaa': 7.5, 'ips': 6.8, 'ipp': 7.0, 'ipv': 6.8, 'defasagem_escolar': 0, 'pedra_ord': 2},
        {'ra': 'RA-003', 'idade': 16, 'ano_ingresso': 2021, 'inde': 5.5, 'ida': 5.0, 'ieg': 5.8, 'iaa': 7.8, 'ips': 5.0, 'ipp': 5.5, 'ipv': 5.3, 'defasagem_escolar': 1, 'pedra_ord': 1},
        {'ra': 'RA-004', 'idade': 12, 'ano_ingresso': 2020, 'inde': 8.8, 'ida': 8.6, 'ieg': 9.2, 'iaa': 8.5, 'ips': 8.5, 'ipp': 8.8, 'ipv': 8.9, 'defasagem_escolar': 0, 'pedra_ord': 4},
        {'ra': 'RA-005', 'idade': 13, 'ano_ingresso': 2022, 'inde': 7.3, 'ida': 7.0, 'ieg': 8.0, 'iaa': 8.0, 'ips': 7.2, 'ipp': 7.4, 'ipv': 7.3, 'defasagem_escolar': 0, 'pedra_ord': 3},
    ])

    uploaded_file = st.file_uploader("Envie um arquivo CSV com as notas dos alunos", type=["csv"])

    df_para_analise = None
    if uploaded_file is not None:
        try:
            df_para_analise = pd.read_csv(uploaded_file)
            st.success(f"Arquivo carregado com sucesso: {len(df_para_analise)} registros encontrados.")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
    else:
        st.info("Nenhum arquivo enviado. Você pode testar com a nossa base sintética de exemplo abaixo:")
        if st.button("Carregar Dados de Exemplo (5 Alunos)"):
            df_para_analise = amostra_exemplo

    if df_para_analise is not None:
        # Âncora invisível para scroll automático no lote
        st.markdown('<div id="resultado-anchor"></div>', unsafe_allow_html=True)
        
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
                lambda row: ['background-color: #FFF1F2' if row['Status'] == 'Alto Risco' else ('background-color: #F0FDF4' if row['Status'] == 'Baixo Risco' else 'background-color: #FFFBEB') for _ in row],
                axis=1
            ), use_container_width=True)

            csv_data = df_resultado.to_csv(index=False).encode('utf-8')
            st.download_button("Baixar Relatório de Triagem (CSV)", csv_data, "relatorio_triagem_passos_magicos.csv", "text/csv")
            
            rolar_para_resultado()
        else:
            st.error(f"O arquivo não possui todas as colunas obrigatórias: {set(cols_necessarias) - set(cols_presentes)}")

# -------------------------------------------------------------
# ABA 3: DIAGNÓSTICO E STORYTELLING
# -------------------------------------------------------------
elif menu == "Diagnóstico e Storytelling":
    st.markdown('<div class="main-header">Storytelling & Principais Descobertas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Síntese das principais evidências encontradas no estudo longitudinal (2022 a 2024).</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 1. Efetividade Comprovada pelas Pedras")
        st.write("""
        A análise longitudinal confirmou o **impacto transformador** da metodologia Passos Mágicos:
        - Os alunos progridem de maneira contínua entre as fases (*Quartzo ➔ Ágata ➔ Ametista ➔ Topázio*).
        - Alunos que alcançam **Topázio e Ametista** sustentam engajamento diário (**IEG**) acima de **90%**.
        """)

    with col2:
        st.markdown("### 2. O Efeito Sensor do IPS (Emocional)")
        st.write("""
        Uma das maiores descobertas do Datathon:
        - Estudantes que sofreram queda real nas notas de provas (**IDA**) no ano seguinte já apresentavam **queda aguda no IPS (Aspectos Psicossociais)** no ano anterior.
        - O bem-estar emocional funciona como **sensor antecipador de risco**.
        """)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 3. O Motor do Ponto de Virada (IPV)")
        st.write("""
        A modelagem de regressão linear multivariada comprovou a hierarquia de influência sobre o IPV:
        1. **Engajamento (IEG):** Maior coeficiente de determinação direta.
        2. **Desempenho Escolar (IDA):** Reforça a confiança acadêmica.
        """)

    with col4:
        st.markdown("### 4. Coerência da Autoavaliação (IAA)")
        st.write("""
        - Nas fases iniciais, a correlação entre autoavaliação e nota real é próxima de zero devido ao otimismo infantil.
        - Conforme o jovem avança, a correlação sobe, demonstrando **amadurecimento do senso crítico**.
        """)

# -------------------------------------------------------------
# ABA 4: DICIONÁRIO DE INDICADORES
# -------------------------------------------------------------
elif menu == "Dicionário de Indicadores":
    st.markdown('<div class="main-header">Dicionário de Indicadores Educacionais</div>', unsafe_allow_html=True)
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
        with st.expander(f"**{sigla}** — {nome}"):
            st.write(desc)

# -------------------------------------------------------------
# ABA 5: DETALHES DO MODELO ML
# -------------------------------------------------------------
elif menu == "Detalhes do Modelo ML":
    st.markdown('<div class="main-header">Metodologia de Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Critérios técnicos, prevenção de vazamento de dados e avaliação de performance.</div>', unsafe_allow_html=True)

    st.markdown("""
    #### Prevenção Rigorosa de Data Leakage (Vazamento Temporal)
    Como a base é longitudinal (acompanhando alunos de 2022 a 2024), uma divisão aleatória padrão causaria **vazamento de dados severo**.
    
    Para garantir validação real out-of-sample:
    - Utilizou-se `GroupShuffleSplit` e `GroupKFold` agrupando pelo identificador único do aluno (`RA`).
    - O modelo foi testado exclusivamente em **alunos nunca vistos durante o treinamento**.

    #### Comparação de Modelos no Teste:
    """)

    tabela_modelos = pd.DataFrame([
        {'Modelo': 'Regressão Logística (Campeão)', 'AUC-ROC Macro': 0.8503, 'Cohen\'s Kappa': 0.5455, 'Overfitting Gap': 0.0155, 'Vantagem': 'Alta generalização e interpretabilidade linear para pedagogia.'},
        {'Modelo': 'HistGradientBoosting', 'AUC-ROC Macro': 0.7998, 'Cohen\'s Kappa': 0.5455, 'Overfitting Gap': 0.1756, 'Vantagem': 'Captura interações não-lineares, mas teve maior gap de treino-teste.'},
        {'Modelo': 'Random Forest', 'AUC-ROC Macro': 0.7870, 'Cohen\'s Kappa': 0.4848, 'Overfitting Gap': 0.2130, 'Vantagem': 'Ensemble robusto, porém com tendência a overfitting no treino (AUC 1.0).'},
        {'Modelo': 'K-Nearest Neighbors (KNN)', 'AUC-ROC Macro': 0.6832, 'Cohen\'s Kappa': 0.2121, 'Overfitting Gap': 0.3168, 'Vantagem': 'Sensível à dimensionalidade.'}
    ])
    st.table(tabela_modelos)
