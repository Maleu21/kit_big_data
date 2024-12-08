import streamlit as st
import pandas as pd


# Configuration de la page
st.set_page_config(page_title="Recipe Dashboard", page_icon="🍲", layout="wide")


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


# Charger les datasets
@st.cache_data
def load_data3(file_path, expected_columns):
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


comparison_columns = [
    "name (RAW)",
    "ingredients (RAW)",
    "generated_name_V1 (RAW)",
    "generated_name (with epirecipes)",
    "% d'ingrédients similaires",
    "title (epirecipes)",
    "ingrédients (epirecipes)",
]

transformed_columns = ["name", "generated_name", "ingredients", "n_ingredients"]
df_comparison = load_data3(
    "second_method_comparison_deployment.csv", comparison_columns
)
df_transformed = load_data3("transformed_recipes_deployment.csv", transformed_columns)

# Titre principal
st.title("Page Bonus : Correction de Noms de Recettes")

# Contexte explicatif
st.markdown(
    """
### Contexte
Nous avons remarqué que les noms des recettes dans le dataset **RAW_recipes.csv** sont souvent peu informatifs ou difficiles à interpréter.  
Pour améliorer cela, nous avons généré de nouveaux noms en utilisant des données issues d'une base externe : 
[EpiRecipes](https://www.kaggle.com/datasets/hugodarwood/epirecipes).  

L'objectif principal était de maximiser la ressemblance statistique entre les variables suivantes dans le dataset initial :
- **name** : nom de la recette,  
- **ingredients** : liste d'ingrédients,  
- **minutes** : temps de préparation (en minutes),  
- **n_ingredients** : nombre d'ingrédients,  
- **steps** : étapes de préparation.  

Les étapes détaillées de ce prétraitement sont documentées dans le fichier **`data_preprocessing.ipynb`**.
"""
)

# Premier dataset : Comparaison des méthodes
st.subheader("Comparaison des Recettes")
st.dataframe(df_comparison)

st.markdown(
    """
**Analyse :**
Ce dataset compare différentes approches utilisées pour générer les nouveaux noms de recettes.  
Les résultats incluent les performances en termes de ressemblance statistique avec les variables originales.
"""
)

# Second dataset : Recettes transformées
st.subheader("Recettes Transformées")
st.dataframe(df_transformed)

st.markdown(
    """
**Analyse :**
Ce dataset contient les recettes transformées avec de nouveaux noms générés.  
Ces noms sont basés sur une analyse de similarité avec des recettes provenant de la base externe **EpiRecipes**.
"""
)

# Lien vers la source de données externe
st.markdown(
    """
Pour en savoir plus sur la base de données externe utilisée, visitez :  
[EpiRecipes Dataset](https://www.kaggle.com/datasets/hugodarwood/epirecipes).

Pour les détails des étapes de prétraitement et d'analyse, référez-vous au fichier :  
**`data_preprocessing.ipynb`**.
"""
)
