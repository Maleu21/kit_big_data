import streamlit as st

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


# Titre principal
st.title("🍲 Recipe Project: Kit Big Data - BGDIA700")
st.markdown(
    "Bienvenue dans le tableau de bord de notre projet **Recettes**. Explorez les données, découvrez les caractéristiques des recettes les plus populaires, et analysez les facteurs influençant les notes des utilisateurs !"
)

# Vue d'ensemble du projet
st.header("📌 Vue d'ensemble du projet")
st.markdown(
    """
Dans le cadre de ce projet, nous utilisons deux ensembles de données riches en informations :
- **RAW_recipes** : Données sur les recettes, y compris le temps de préparation, les ingrédients et les étapes.
- **RAW_interactions** : Données sur les interactions des utilisateurs avec les recettes, comme les notes et les avis.

**Problématique principale :**
1. Quelles sont les caractéristiques des recettes les plus populaires ?
2. Quels sont les facteurs influençant les notes des recettes, en prenant en compte :
   - Le temps de préparation,
   - Le nombre d'étapes,
   - Les ingrédients utilisés ?
3-Visualisation des recettes par réduction de dimension pour déterminer les
recettes qui se rapprochent de celles que l'on a déjà faites.
4-Est-ce que les utilisateurs les plus anciens obtiennent de meilleures notes à leurs
recettes ?
"""
)

# Présentation des colonnes des datasets
st.header("📊 Description des colonnes disponibles")
with st.expander("Voir la description des colonnes de RAW_recipes"):
    st.markdown(
        """
    **RAW_recipes**
    - `id` : Identifiant unique de la recette.
    - `name` : Nom de la recette.
    - `minutes` : Temps total de préparation (en minutes).
    - `contributor_id` : Identifiant de l'utilisateur ayant contribué à la recette.
    - `submitted` : Date de soumission de la recette.
    - `tags` : Liste de tags associés à la recette (par exemple, "végétarien", "facile").
    - `nutrition` : Informations nutritionnelles (calories, graisses, sucres, etc.).
    - `n_steps` : Nombre d'étapes nécessaires.
    - `steps` : Liste des étapes de préparation.
    - `description` : Brève description de la recette.
    - `ingredients` : Liste des ingrédients nécessaires.
    - `n_ingredients` : Nombre total d'ingrédients requis.
    """
    )

with st.expander("Voir la description des colonnes de RAW_interactions"):
    st.markdown(
        """
    **RAW_interactions**
    - `user_id` : Identifiant unique de l'utilisateur.
    - `recipe_id` : Identifiant unique de la recette.
    - `date` : Date de l'interaction.
    - `rating` : Note attribuée par l'utilisateur (sur une échelle de 1 à 5).
    - `review` : Commentaire de l'utilisateur sur la recette.
    """
    )

# Focus sur les questions de recherche
st.header("🔍 Questions de recherche")
st.markdown(
    """
Pour répondre à nos problématiques, nous combinerons les informations des deux datasets pour analyser :
1. **Les caractéristiques des recettes les plus populaires**, basées sur leur nombre d'interactions et leur note moyenne.
2. **Les facteurs influençant les notes**, en explorant des variables comme le temps de préparation, le nombre d'étapes, et le nombre d'ingrédients.

---

### 🏆 Point Bonus :
Nous irons plus loin en utilisant le **NLP** pour générer des **noms de recettes à la fois créatifs et logiques** à partir des données disponibles.
"""
)

# Sidebar navigation
st.sidebar.success("Naviguez entre les pages pour découvrir les analyses !")


# Ajouter des pages supplémentaires
st.sidebar.title("Navigation")
st.sidebar.markdown("Utilisez le menu ci-dessus pour accéder aux différentes analyses :")

st.sidebar.title("Data Options")
st.sidebar.markdown("**Upload Recipe Dataset (CSV file)**")
st.sidebar.caption("Limit: 1GB per file")

uploaded_files = st.sidebar.file_uploader(
    "Drag and drop file here",
    type="csv",
    accept_multiple_files=True,
    help="You can upload multiple CSV files up to 1GB each."
)

if uploaded_files:
    st.success("Files uploaded successfully!")
    for file in uploaded_files:
        st.write(f"Uploaded file: {file.name}")

st.sidebar.write("- [Réduction de dimension](Reduction_de_dimension_app.py)")
st.sidebar.write("- [Génération de noms](generated_name_app.py)")

st.sidebar.success("Naviguez entre les pages pour découvrir les analyses !")

# Configuration pour la taille du dataset
st.sidebar.markdown("**Limite actuelle de chargement des fichiers : 1GB**")

