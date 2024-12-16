def plot_disease_distribution(data):
    """Affiche la distribution des maladies dans les données"""
    count_disease = data["prognosis"].value_counts()
    temp_df = pd.DataFrame({
        "Maladies": count_disease.index,
        "Nombre": count_disease.values
    })
    plt.figure(figsize=(18,8))
    sns.barplot(x="Maladies", y="Nombre", data=temp_df)
    plt.xticks(rotation=90)
    plt.show()

def encode_target(data):
    """Encode la cible 'prognosis' en valeurs numériques"""
    encoding = LabelEncoder()
    data["prognosis"] = encoding.fit_transform(data["prognosis"])
    return data, encoding