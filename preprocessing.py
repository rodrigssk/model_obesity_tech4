def preprocess_data(df):
    df = df.copy()

    # Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Ordinais
    ordinal_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df['Consumption of food between meals'] = df['Consumption of food between meals'].map(ordinal_map)
    df['Alcohol consumption'] = df['Alcohol consumption'].map(ordinal_map)

    # Transporte
    transport_map = {
        'Walking': 1,
        'Bike': 2,
        'Public_Transportation': 3,
        'Motorbike': 4,
        'Automobile': 5
    }
    df['Transportation used'] = df['Transportation used'].map(transport_map)

    return df
