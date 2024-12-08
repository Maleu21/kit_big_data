import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Configuration de la page
st.set_page_config(page_title="Recipe Dashboard", page_icon="🍲", layout="wide")
st.title("🍲 Popular Recipes")

<<<<<<< HEAD
=======
# Ajout d'un fond noir et personnalisation de la barre latérale avec une ombre blanche
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), url("https://media.cdnws.com/_i/96967/25307/3948/10/deco-table-avec-assiette-noire.jpeg") no-repeat center center;
        background-size: cover;
        color: black;
    }
    .stApp {
        background: linear-gradient(90deg, gray, black);
        color: white;
    }
    section[data-testid="stSidebar"] .css-1d391kg {
        color: black;
    }
    section[data-testid="stSidebar"] .css-18e3th9 {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)
>>>>>>> Nohaila2

@st.cache_data
def load_data2(file_path, expected_columns):
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
    "contributor_id",
    "submitted",
    "tags",
    "nutrition",
    "n_steps",
    "steps",
    "description",
    "ingredients",
    "n_ingredients",
    "average_rating",
    "num_interactions",
    "prep_time_category",
]


df_popular_recipes = load_data2("popular_recipes.csv", recipes_columns)

# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Recette populaire:</h3>
        <p>On a fait le seuil du troisième quartile (quantile 0.75) : 4.0. On a pris 
            que les recettes qui ont plus de 4 interactions.</p>
    </div>
""",
    unsafe_allow_html=True,
)

st.write("Voici les données chargées :")
st.dataframe(df_popular_recipes.head(10))

# Ajouter un titre à la page Streamlit
st.title("Distribution des Notes Moyennes")

# Créer un histogramme avec Plotly Express
fig = px.histogram(
    df_popular_recipes,
    x="average_rating",  # Colonne à afficher sur l'axe x
    nbins=20,  # Nombre de bins
    title="Distribution des notes moyennes des recettes",
    labels={"average_rating": "Note moyenne"},
    color_discrete_sequence=["orange"],  # Couleur de l'histogramme
)

# Ajouter les labels pour l'axe y
fig.update_layout(xaxis_title="Note moyenne", yaxis_title="Nombre de recettes")
st.plotly_chart(fig)


# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique :</h3>
        <p>On constate d'après ce graphique qu'on élimine beaucoup de recettes ayant obtenu 
            la note 5. C'est pour cela qu'on voit que la bin de la note 5 est devenue plus 
            proche des deux autres bins qui se trouvent entre la note 4 et 5. On voit aussi 
            qu'on a moins de recettes ayant obtenu une note inférieure à 4</p>
    </div>
""",
    unsafe_allow_html=True,
)

st.write("Statistiques descriptives de la dataset:")
st.dataframe(df_popular_recipes.describe())

# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique :</h3>
        <p>On a fait cette description pour ce dataset afin de vérifier notre analyse pour 
        le premier graphique, et on constate que la description valide notre point de vue, 
        car on observe une augmentation de la moyenne des notes pour ce dataset avec ces quartiles.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Calculer la matrice de corrélation
correlation_matrix = df_popular_recipes[
    ["minutes", "n_ingredients", "n_steps", "average_rating", "num_interactions"]
].corr()

# Créer le graphique heatmap avec Plotly Express
fig = px.imshow(
    correlation_matrix,
    text_auto=True,  # Afficher les valeurs dans les cases
    color_continuous_scale="RdBu_r",  # Palette de couleurs
    title="Matrice de Corrélation",
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)

# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du Matrice de Corrélation :</h3>
        <p>D'après la matrice de corrélation, on constate qu'il y a une corrélation positive 
            entre le nombre d'étapes et le nombre d'ingrédients, et de même pour les minutes. 
            Il y a aussi une corrélation négative entre les minutes et la note moyenne. C'est tout 
            à fait logique.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Créer un histogramme avec Plotly Express (et un KDE en même temps)
fig = px.histogram(
    df_popular_recipes,
    x="n_steps",
    nbins=10,
    marginal="rug",  # Ajouter un "rug plot" pour la densité
    title="Distribution du Nombre d'Étapes des Recettes Populaires",
    labels={"n_steps": "Nombre d'Étapes"},
    opacity=0.7,
    color_discrete_sequence=["orange"],
)

# Afficher le graphique dans Streamlit
st.plotly_chart(fig)

# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique  :</h3>
        <p>On constate que presque toutes les recettes populaires ont entre 5 et 10 étapes au maximum.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# 2. Vectorisation TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_popular_recipes["ingredients"])

# 3. Affichage des mots clés les plus importants (Ingrédients)
feature_names = tfidf_vectorizer.get_feature_names_out()
important_features = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names).sum().sort_values(ascending=False)  # type: ignore

# Sélectionner les 7 ingrédients les plus importants
top_7_ingredients = important_features.head(7)

# Afficher les 7 ingrédients les plus importants dans Streamlit
st.write("Les 7 ingrédients les plus importants dans le dataset (basé sur TF-IDF) :")
st.write(top_7_ingredients)

# 4. Création du plot des 7 ingrédients les plus importants
fig_ingredients = px.bar(
    top_7_ingredients,
    x=top_7_ingredients.index,
    y=top_7_ingredients.values,
    title="Les 7 Ingrédients les Plus Importants basés sur TF-IDF",
    labels={"x": "Ingrédients", "y": "Importance (TF-IDF)"},
)

fig_ingredients.update_layout(
    xaxis_title="Ingrédients", yaxis_title="Importance (TF-IDF)", width=1000, height=600
)

# Afficher le plot dans Streamlit
st.plotly_chart(fig_ingredients)


# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique  :</h3>
        <p>Ce sont les ingrédients de base pour chaque recette. C'est très logique 
            qu'on les voie très fréquemment dans toutes les recettes.</p>
    </div>
""",
    unsafe_allow_html=True,
)

top_500_recipes = df_popular_recipes.sort_values(
    by=["average_rating", "num_interactions"], ascending=[False, False]
).head(500)

worst_500_recipes = df_popular_recipes.sort_values(
    by=["average_rating", "num_interactions"], ascending=[True, False]
).head(500)

# Analyser les ingrédients des recettes populaires
top_500_popular_ingredients = Counter(
    ingredient
    for ingredients in top_500_recipes["ingredients"]
    for ingredient in eval(ingredients)
)


# Créer une fonction pour tracer le graphique avec un nombre d'ingrédients variable
def plot_top_ingredients(n):
    most_common_popular = top_500_popular_ingredients.most_common(n)

    # Convertir en DataFrame
    popular_ingredients_df = pd.DataFrame(
        most_common_popular, columns=["Ingredient", "Count"]
    )

    # Création du graphique avec Plotly
    fig = px.bar(
        popular_ingredients_df,
        x="Count",
        y="Ingredient",
        orientation="h",
        title=f"Les {n} Ingrédients les Plus Fréquents dans les Recettes Populaires",
        labels={"Count": "Nombre d’apparitions", "Ingredient": "Ingrédients"},
    )

    fig.update_layout(width=1000, height=600)

    return fig


# Ajouter un slider pour sélectionner le nombre d'ingrédients à afficher
num_ingredients = st.slider(
    "Sélectionner le nombre d'ingrédients à afficher", 5, 20, 10, key="unique_slider_1"
)

# Afficher le graphique en fonction de la sélection du slider
st.plotly_chart(plot_top_ingredients(num_ingredients))

# Afficher l'analyse dans une boîte de style
st.markdown(
    """
    <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 5px;">
        <h3>Analyse du graphique  :</h3>
        <p>Comme l'autre graphique, il y a des ingrédients qui sont nécessaires pour 
            chaque recette. Mais cette fois, on voit que l'ingrédient 'olive oil', qui 
            apparaît en 4e position dans les recettes les mieux notées, n'est cependant pas présent 
            parmi les 7 ingrédients les plus utilisés.</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Analyser les ingrédients des recettes populaires
worst_500_popular_ingredients = Counter(
    ingredient
    for ingredients in worst_500_recipes["ingredients"]
    for ingredient in eval(ingredients)
)


# Créer une fonction pour tracer le graphique avec un nombre d'ingrédients variable
def plot_worst_ingredients(n):
    most_common_popular = worst_500_popular_ingredients.most_common(n)

    # Convertir en DataFrame
    popular_ingredients_df = pd.DataFrame(
        most_common_popular, columns=["Ingredient", "Count"]
    )

    # Création du graphique avec Plotly
    fig = px.bar(
        popular_ingredients_df,
        x="Count",
        y="Ingredient",
        orientation="h",
        title=f"Les {n} Ingrédients les Plus Fréquents dans les Recettes Populaires",
        labels={"Count": "Nombre d’apparitions", "Ingredient": "Ingrédients"},
    )

    fig.update_layout(width=1000, height=600)

    return fig


# Ajouter un slider pour sélectionner le nombre d'ingrédients à afficher
num_ingredients2 = st.slider(
    "Sélectionner le nombre d'ingrédients à afficher", 5, 20, 10, key="unique_slider_2"
)

# Afficher le graphique en fonction de la sélection du slider
st.plotly_chart(plot_worst_ingredients(num_ingredients2))
