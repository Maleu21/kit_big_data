import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Recipe Dashboard", page_icon="🍲", layout="wide")

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
