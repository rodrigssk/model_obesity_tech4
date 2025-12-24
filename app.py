# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import joblib

# IMPORTANTE: necessário para o joblib reconstruir a pipeline
from preprocessing import preprocess_data  # noqa: F401

# ===============================
# CONFIGURAÇÃO DA PÁGINA
# ===============================
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🧠",
    layout="centered"
)

# ===============================
# ESTILO
# ===============================
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f4f6f9;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# CARREGAR PIPELINE E ENCODER
# ===============================
pipeline = joblib.load("modelo_obesidade_pipeline.pkl")
le_target = joblib.load("label_encoder.pkl")

# ===============================
# DICIONÁRIOS DE TRADUÇÃO (UI → MODELO)
# ===============================
gender_map = {
    "Masculino": "Male",
    "Feminino": "Female"
}

yes_no_map = {
    "Sim": "yes",
    "Não": "no"
}

ordinal_map = {
    "Não consome": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always"
}

transport_map = {
    "A pé": "Walking",
    "Bicicleta": "Bike",
    "Transporte público": "Public_Transportation",
    "Motocicleta": "Motorbike",
    "Automóvel": "Automobile"
}

# ===============================
# TRADUÇÃO DAS CLASSES (MODELO → UI)
# ===============================
target_translation = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso – Grau I",
    "Overweight_Level_II": "Sobrepeso – Grau II",
    "Obesity_Type_I": "Obesidade – Grau I",
    "Obesity_Type_II": "Obesidade – Grau II",
    "Obesity_Type_III": "Obesidade – Grau III"
}

# ===============================
# TÍTULO
# ===============================
st.title("🧠 Predição de Obesidade")
st.markdown(
    "Este aplicativo auxilia profissionais de saúde na **classificação do nível de obesidade**, "
    "utilizando informações antropométricas e hábitos de vida como **apoio à decisão clínica**."
)

st.divider()

# ===============================
# FORMULÁRIO
# ===============================
with st.form("formulario_obesidade"):
    st.subheader("📋 Informações Pessoais")

    col1, col2 = st.columns(2)
    with col1:
        gender_label = st.selectbox("Gênero", list(gender_map.keys()))
        age = st.number_input("Idade", min_value=10, max_value=100, value=30)

    with col2:
        height = st.number_input("Altura (m)", min_value=1.30, max_value=2.20, value=1.70)
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)

    st.subheader("🧬 Histórico e Hábitos")

    family_label = st.selectbox(
        "Possui histórico familiar de obesidade?",
        list(yes_no_map.keys())
    )

    high_caloric_label = st.selectbox(
        "Consome frequentemente alimentos altamente calóricos?",
        list(yes_no_map.keys())
    )

    smoke_label = st.selectbox(
        "Você fuma?",
        list(yes_no_map.keys())
    )

    calories_monitor_label = st.selectbox(
        "Você monitora o consumo de calorias?",
        list(yes_no_map.keys())
    )

    st.subheader("🍎 Hábitos Alimentares")

    vegetables = st.slider(
        "Frequência de consumo de vegetais (0 = nunca, 3 = sempre)",
        min_value=0, max_value=3, value=2
    )

    meals = st.slider(
        "Número de refeições principais por dia",
        min_value=1, max_value=5, value=3
    )

    snacks_label = st.selectbox(
        "Consumo de alimentos entre as refeições",
        list(ordinal_map.keys())
    )

    alcohol_label = st.selectbox(
        "Consumo de álcool",
        list(ordinal_map.keys())
    )

    st.subheader("🏃 Estilo de Vida")

    physical_activity = st.slider(
        "Frequência de atividade física (0 = nunca, 3 = sempre)",
        min_value=0, max_value=3, value=1
    )

    screen_time = st.slider(
        "Tempo diário em dispositivos eletrônicos (horas)",
        min_value=0, max_value=10, value=4
    )

    water = st.slider(
        "Consumo diário de água (litros)",
        min_value=0.5, max_value=5.0, value=2.0
    )

    transport_label = st.selectbox(
        "Meio de transporte principal",
        list(transport_map.keys())
    )

    submitted = st.form_submit_button("🔍 Analisar Perfil")

# ===============================
# PREDIÇÃO
# ===============================
if submitted:
    input_data = pd.DataFrame([{
        'Gender': gender_map[gender_label],
        'Age': age,
        'family_history': 1 if yes_no_map[family_label] == "yes" else 0,
        'Frequent consumption of high-caloric food': 1 if yes_no_map[high_caloric_label] == "yes" else 0,
        'SMOKE': 1 if yes_no_map[smoke_label] == "yes" else 0,
        'Calories consumption monitoring': 1 if yes_no_map[calories_monitor_label] == "yes" else 0,
        'Frequency of consumption of vegetables': vegetables,
        'Number of main meals': meals,
        'Consumption of food between meals': ordinal_map[snacks_label],
        'Physical activity frequency': physical_activity,
        'Time using electronic devices': screen_time,
        'Daily water consumption': water,
        'Alcohol consumption': ordinal_map[alcohol_label],
        'Transportation used': transport_map[transport_label],
        'Height': height,
        'Weight': weight
    }])

    # Garantir mesmas features e mesma ordem do treino
    input_data = input_data.reindex(columns=pipeline.feature_names_in_)

    # Predição
    prediction = pipeline.predict(input_data)
    raw_result = le_target.inverse_transform(prediction)[0]
    result_pt = target_translation.get(raw_result, raw_result)

    # ===============================
    # RESULTADOS
    # ===============================
    st.success("✅ Análise concluída!")

    st.metric(
        "🧠 Classificação predita",
        result_pt
    )

    st.info(
        "ℹ️ Esta predição é uma **ferramenta de apoio à decisão clínica** e "
        "não substitui a avaliação médica individual."
    )
