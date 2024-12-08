import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px


# Configuration de la page
st.set_page_config(page_title="Recipe Dashboard", page_icon="🍲", layout="wide")
st.title("🍲 All Recipes")


@st.cache_data
def load_data(file_path, expected_columns):
    """
    Charge un fichier CSV et vérifie les colonnes.

    Args:
        file_path (str): Chemin vers le fichier CSV.
        expected_columns (list): Liste des colonnes attendues.

    Returns:
        pd.DataFrame: DataFrame contenant les données du fichier CSV.
    """
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in expected_columns):
            st.error(f"Colonnes manquantes dans le fichier : {file_path}")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error(f"Fichier non trouvé : {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return pd.DataFrame()


# Chargement des fichiers avec les colonnes attendues
recipes_columns = [
    "name",
    "id",
    "minutes",
    "tags",
    "n_steps",
    "ingredients",
    "n_ingredients",
]
interactions_columns = ["recipe_id", "rating"]

recipes_df = load_data("recipes.csv", recipes_columns)
interactions_df = load_data("interactions.csv", interactions_columns)
recipes_df = pd.read_csv("recipes.csv")
interactions_df = pd.read_csv("interactions.csv.csv")
# Titre de la section
st.title("Distribution du Temps de Préparation des Recettes")

# Créer un graphique interactif avec Plotly Express pour afficher la distribution des temps de préparation
fig = px.histogram(
    recipes_df,  # Dataset
    x="minutes",  # Colonne pour l'axe des x (temps de préparation)
    nbins=30,  # Nombre de bins pour l'histogramme
    title="Distribution du Temps de Préparation des Recettes",  # Titre
    labels={
        "minutes": "Temps de Préparation (minutes)",
        "count": "Fréquence",
    },  # Labels pour les axes
    color_discrete_sequence=["teal"],  # Couleur de l'histogramme
    opacity=0.75,  # Transparence
)

# Ajuster la mise en page du graphique
fig.update_layout(
    xaxis_title="Temps de Préparation (minutes)",
    yaxis_title="Fréquence",
    bargap=0.1,  # Espacement entre les barres
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)

# Analyse du graphique dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique :</h3>
        <p>Ce graphique montre la distribution des temps de préparation des recettes. Nous observons que 
        la majorité des recettes nécessitent un temps de préparation court. Cependant, il existe des valeurs aberrantes où les temps de préparation
        sont extrêmement longs. Ces valeurs peuvent fausser l'analyse globale.</p>
        <p>Pour une analyse plus précise, il est conseillé d'éliminer ces valeurs aberrantes et de se concentrer 
        sur les recettes ayant un temps de préparation raisonnable.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Titre de la section
st.title("Temps de Préparation des Recettes")

# Créer un boxplot interactif avec Plotly Express
fig = px.box(
    recipes_df,  # Dataset
    x="minutes",  # Colonne pour l'axe des x
    title="Boxplot du Temps de Préparation des Recettes",  # Titre
    labels={"minutes": "Temps de Préparation (minutes)"},  # Labels pour les axes
    color_discrete_sequence=["royalblue"],  # Couleur du boxplot
)

# Ajuster la mise en page du graphique
fig.update_layout(
    xaxis_title="Temps de Préparation (minutes)",
    yaxis_title="",  # Pas de label sur l'axe Y car ce n'est pas nécessaire ici
    showlegend=False,  # Pas de légende pour un seul boxplot
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)

# Analyse du boxplot dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #FF5722; padding: 15px; border-radius: 5px;">
        <h3>Analyse du boxplot :</h3>
        <p>Ce boxplot montre que la majorité des temps de préparation sont concentrés dans une plage raisonnable,
        mais il existe de valeurs aberrantes, avec des temps de préparation très longs.</p>
        <p>Pour exploiter correctement le dataset et éviter que ces outliers influencent les résultats, 
        il est recommandé d'appliquer un filtre pour exclure ces valeurs extrêmes.</p>
    </div>
""",
    unsafe_allow_html=True,
)


def clean_outliers(df, column):
    """Supprime les outliers d'une colonne basée sur l'IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


recipes_df_cleaned = clean_outliers(recipes_df, "minutes")


# Notification pour expliquer le traitement des outliers
st.info(
    """
    **Notice :**  
    Pour éliminer les valeurs aberrantes, on'ai utilisé la méthode de l'intervalle interquartile (IQR). 
    Cette méthode consiste à calculer le premier quartile (Q1) et le troisième quartile (Q3) d'une variable. 
    Ensuite, on détermine l'IQR en soustrayant Q1 de Q3. Les valeurs qui se situent en dehors des limites 
    définies par 1,5 fois l'IQR sont considérées comme des outliers. Cette approche permet d'identifier et de 
    supprimer les valeurs extrêmes afin d'améliorer l'analyse des données.
"""
)
# Titre de la section
st.title("Temps de Préparation des Recettes apres la suppression des outliers ")

# Créer un boxplot interactif avec Plotly Express
fig = px.box(
    recipes_df_cleaned,  # Dataset
    x="minutes",  # Colonne pour l'axe des x
    title="Boxplot du Temps de Préparation des Recettes",  # Titre
    labels={"minutes": "Temps de Préparation (minutes)"},  # Labels pour les axes
    color_discrete_sequence=["royalblue"],  # Couleur du boxplot
)

# Ajuster la mise en page du graphique
fig.update_layout(
    xaxis_title="Temps de Préparation (minutes)",
    yaxis_title="",  # Pas de label sur l'axe Y car ce n'est pas nécessaire ici
    showlegend=False,  # Pas de légende pour un seul boxplot
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)

# Analyse détaillée du boxplot
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du Boxplot :</h3>
        <p>Ce boxplot montre la répartition des temps de préparation des recettes après nettoyage des données.</p>
        <ul>
            <li><strong>Premier quartile (Q1) :</strong> Environ 20 minutes. Cela signifie que 25 % des recettes nécessitent moins de 20 minutes pour être préparées.</li>
            <li><strong>Médiane (Q2) :</strong> Environ 35 minutes. La moitié des recettes prennent moins de 35 minutes.</li>
            <li><strong>Troisième quartile (Q3) :</strong> Environ 55 minutes. 75 % des recettes nécessitent moins de 55 minutes.</li>
            <li><strong>Valeurs extrêmes :</strong> La majorité des outliers ont été éliminés. Les temps de préparation varient maintenant de 0 minute (minimum) à 105 minutes (maximum), ce qui est réaliste.</li>
        </ul>
        <p>Ce nettoyage des données permet une meilleure analyse, en supprimant les valeurs aberrantes qui faussaient la représentation globale. Les temps de préparation sont concentrés autour de 20 à 55 minutes, ce qui reflète des recettes généralement accessibles et rapides à réaliser.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Notification pour expliquer le traitement des outliers
st.info(
    """
    **Notice :**  
    Nous avons fait la même chose pour n_steps et n_ingredients.
"""
)
recipes_df_cleaned = clean_outliers(recipes_df_cleaned, "n_steps")
recipes_df_cleaned = clean_outliers(recipes_df_cleaned, "n_ingredients")

# Adjusting the merge to account for different column names
merged_data = pd.merge(
    recipes_df_cleaned, interactions_df, left_on="id", right_on="recipe_id"
)
##Drop the redundant 'recipe_id' column from the merged DataFrame because we have two columns with the same information (recipe_id and id).
merged_data = merged_data.drop(columns=["recipe_id"])
recipe_ratings = merged_data.groupby("id")["rating"].mean().reset_index()
recipe_ratings.columns = ["id", "average_rating"]

# Ajouter la note moyenne au dataset nettoyé
final_data = pd.merge(recipes_df_cleaned, recipe_ratings, on="id")
interaction_counts = merged_data.groupby("id")["rating"].count().reset_index()
interaction_counts.columns = ["id", "num_interactions"]
# Ajouter le nombre d'interactions au dataset final
final_data = pd.merge(final_data, interaction_counts, on="id")
# créer des catégories pour le temps de préparation
final_data["prep_time_category"] = pd.cut(
    final_data["minutes"],
    bins=[0, 30, 60, 120, float("inf")],
    labels=["<30 min", "30-60 min", "1-2 hrs", ">2 hrs"],
)
# Supprimer les lignes contenant des valeurs manquantes
final_data_reduced = final_data.dropna()


df = final_data_reduced

st.write("Voici les données chargées :")
st.dataframe(df.head(10))

# Ajouter un titre à la page Streamlit
st.title("Distribution des Notes Moyennes")

# Créer un histogramme avec Plotly Express
fig = px.histogram(
    df,
    x="average_rating",  # Colonne à afficher sur l'axe x
    nbins=20,  # Nombre de bins
    title="Distribution des notes moyennes des recettes",
    labels={"average_rating": "Note moyenne"},
    color_discrete_sequence=["purple"],  # Couleur de l'histogramme
)

# Ajouter les labels pour l'axe y
fig.update_layout(xaxis_title="Note moyenne", yaxis_title="Nombre de recettes")
st.plotly_chart(fig)


# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique :</h3>
        <p>la majorité des notes sont de 5. On trouve également des notes de 0 et de 4, mais 
            elles sont beaucoup moins fréquentes par rapport à la note de 5.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Titre
st.title("Average Rating vs. Number of Interactions")

# Créer un graphique interactif avec Plotly
fig = px.scatter(
    df,
    x="num_interactions",
    y="average_rating",
    color="prep_time_category",
    title="Average Rating vs. Number of Interactions",
    labels={
        "num_interactions": "Number of Interactions",
        "average_rating": "Average Rating",
    },
    hover_data=df.columns,  # Afficher toutes les colonnes au survol
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)

# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique :</h3>
        <p>On ne peut pas extraire de grandes conclusions à partir de ce plot, mais ce qui est remarquable, 
        c'est que nous ne trouvons pas de points rouges dans le graphique, représentant les recettes avec 
        un temps de préparation de plus de 2 heures, proches de la note 5. De plus, nous remarquons 
        que les recettes ayant un nombre d'interactions élevé se retrouvent dans toutes les catégories, 
        à l'exception de celles de plus de 2 heures, ce qui est très logique.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Compter les occurrences de chaque catégorie dans 'prep_time_category'
prep_time_counts = df["prep_time_category"].value_counts().reset_index()

# Renommer les colonnes pour une meilleure clarté
prep_time_counts.columns = ["prep_time_category", "count"]

# Créer un graphique en barres avec plotly.express
fig = px.bar(
    prep_time_counts,  # Utiliser le DataFrame avec les données comptées
    x="prep_time_category",  # Colonne pour l'axe X
    y="count",  # Colonne pour l'axe Y
    labels={
        "prep_time_category": "Preparation Time Category",
        "count": "Number of Recipes",
    },
    title="Number of Recipes by Preparation Time Category",
    color="prep_time_category",  # Optionnel : ajouter des couleurs par catégorie
)
st.plotly_chart(fig)


# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique :</h3>
        <p>D'après ce diagramme à barres, nous observons que le nombre de recettes ayant un temps de 
            préparation de moins de 30 minutes est significativement plus élevé que celui des recettes 
            préparées en 30 à 60 minutes. De plus, le nombre de recettes dans la catégorie de 30 à 60 minutes
            dépasse également celui des recettes nécessitant entre 1 et 2 heures. Enfin, les recettes dont 
            le temps de préparation dépasse 2 heures sont les moins nombreuses.Cela suggère que les 
            utilisateurs semblent préférer des recettes rapides et accessibles.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Calculer la moyenne des notes pour chaque catégorie de temps de préparation
average_rating_per_category = (
    df.groupby("prep_time_category")["average_rating"].mean().reset_index()
)

# Créer un graphique en barres avec Plotly Express
fig = px.bar(
    average_rating_per_category,  # Données pour le graphique
    x="prep_time_category",  # Catégorie de temps de préparation sur l'axe X
    y="average_rating",  # Note moyenne sur l'axe Y
    labels={
        "prep_time_category": "Preparation Time Category",
        "average_rating": "Average Rating",
    },
    title="Average Rating per Preparation Time Category",
    color="prep_time_category",  # Optionnel : Ajout de couleurs par catégorie de temps
    color_continuous_scale="Viridis",  # Palette de couleurs pour le graphique
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)

st.write("Statistiques descriptives de la dataset:")
st.dataframe(df.describe())

# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique :</h3>
        <p>D'après le graphique et la description du dataset, nous constatons que la note moyenne 
            de chaque recette est très élevée, avec une moyenne de 4,3 et un premier quartile à 4. 
            Par conséquent, nous allons nous concentrer sur le nombre d'interactions. Nous définirons 
            un seuil pour identifier les recettes populaires, en considérant ici celles qui ont un 
            nombre d'interactions supérieur au troisième quartile. Ainsi, nous ne conserverons que les 
            recettes ayant un minimum de 4 interactions.</p>
    </div>
""",
    unsafe_allow_html=True,
)
