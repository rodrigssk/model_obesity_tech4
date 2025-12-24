# ===============================
# Básico
# ===============================
import pandas as pd
import numpy as np

# ===============================
# Visualização
# ===============================
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Streamlit
# ===============================
import streamlit as st

# ===============================
# Scikit-learn – Pré-processamento e Pipeline
# ===============================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer

# ===============================
# Scikit-learn – Modelagem
# ===============================
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# ===============================
# Scikit-learn – Avaliação e Split
# ===============================
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# Salvar o modelo
# ===============================
import joblib

from preprocessing import preprocess_data
from sklearn.preprocessing import FunctionTransformer

## Carregando base Obesity

df = pd.read_csv('Obesity.csv')

df['family_history'] = df['family_history'].replace(['yes', 'no'],[1,0])
df['FAVC'] = df['FAVC'].replace(['yes', 'no'],[1,0])
df['SMOKE'] = df['SMOKE'].replace(['yes', 'no'],[1,0])
df['SCC'] = df['SCC'].replace(['yes', 'no'],[1,0])

df = df.rename(columns={'FAVC':'Frequent consumption of high-caloric food','FCVC':'Frequency of consumption of vegetables',
                      'NCP':'Number of main meals','CAEC':'Consumption of food between meals',
                       'CH2O':'Daily water consumption','SCC':'Calories consumption monitoring',
                      'FAF':'Physical activity frequency','TUE':'Time using electronic devices',
                      'CALC':'Alcohol consumption','MTRANS':'Transportation used'})


df = df[['Gender','family_history','Frequent consumption of high-caloric food','SMOKE','Calories consumption monitoring',
                                            'Age','Height','Frequency of consumption of vegetables','Weight','Number of main meals','Daily water consumption',
                                             'Physical activity frequency','Time using electronic devices',
                                            'Consumption of food between meals','Alcohol consumption','Transportation used','Obesity']]

# ===============================
# FUNÇÃO DE PRÉ-PROCESSAMENTO
# ===============================
def preprocess_data(df):
    df = df.copy()

    # Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Ordinais
    ordinal_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df['Consumption of food between meals'] = df['Consumption of food between meals'].map(ordinal_map)
    df['Alcohol consumption'] = df['Alcohol consumption'].map(ordinal_map)

    # Transportation (granular)
    transport_map = {
        'Walking': 1,
        'Bike': 2,
        'Public_Transportation': 3,
        'Motorbike': 4,
        'Automobile': 5
    }
    df['Transportation used'] = df['Transportation used'].map(transport_map)

    return df


# ===============================
# DADOS
# ===============================
X = df.drop('Obesity', axis=1)
y_raw = df['Obesity']

# Target encoder
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

# ===============================
# PIPELINE
# ===============================
pipeline = Pipeline(steps=[
    ('preprocess', FunctionTransformer(preprocess_data)),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    ))
])

# ===============================
# SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# TREINO
# ===============================
pipeline.fit(X_train, y_train)

# ===============================
# AVALIAÇÃO
# ===============================
y_pred = pipeline.predict(X_test)

joblib.dump(pipeline, 'modelo_obesidade_pipeline.pkl')
joblib.dump(le_target, 'label_encoder.pkl')