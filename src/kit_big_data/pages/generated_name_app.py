import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import matplotlib.pyplot as plt
import re

# Initialize NLTK
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

# Explicitly set NLTK data path if needed
nltk.data.path.append(r'C:\Users\user\nltk_data')  # Adjust to your system's nltk_data path

stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

st.title("Recipe Processing and Analysis")

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

# File upload
uploaded_files = st.sidebar.file_uploader("Upload Recipe Dataset(s) (CSV)", type="csv", accept_multiple_files=True)

if uploaded_files:
    datasets = {}
    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            datasets[uploaded_file.name] = df
            st.success(f"Loaded {uploaded_file.name} successfully!")
        except Exception as e:
            st.error(f"Failed to load {uploaded_file.name}: {e}")

    if datasets:
        st.header("Datasets Overview")
        for name, df in datasets.items():
            st.subheader(f"Dataset: {name}")
            st.write(df.head())

        # Process a specific dataset (assuming RAW_recipes.csv as an example)
        if 'RAW_recipes.csv' in datasets:
            RAW_recipe = datasets['RAW_recipes.csv']

            # EDA
            st.header("Exploratory Data Analysis")

            total_recipes_count = RAW_recipe.shape[0]
            zero_or_null_minutes_count = RAW_recipe[RAW_recipe['minutes'].isnull() | (RAW_recipe['minutes'] == 0)].shape[0]

            st.write(f"**Total Recipes:** {total_recipes_count}")
            st.write(f"**Recipes with zero or null cooking time:** {zero_or_null_minutes_count}")

            # Filter dataset for useful columns
            raw_filtered = RAW_recipe[['name', 'ingredients', 'steps']]
            st.subheader("Filtered RAW Recipes")
            st.write(raw_filtered.head())

            # Filtering Recipes
            st.header("Filter Recipes")

            max_minutes = st.slider("Maximum Cooking Time (Minutes)", 0, 120, 60)
            filtered_recipes = RAW_recipe[RAW_recipe['minutes'] <= max_minutes]
            st.write(f"Filtered Recipes (Cooking Time <= {max_minutes} minutes):")
            st.write(filtered_recipes[['name', 'minutes', 'ingredients']].head())

            # Recipe Name Generation
            st.header("Generate Recipe Names")

            def extract_cooking_techniques(steps):
                techniques = ['bake', 'fry', 'grill', 'roast', 'boil', 'simmer', 'steam', 'sauté']
                found_techniques = []
                tokens = re.findall(r'\b\w+\b', str(steps).lower())  # Simple tokenizer as fallback
                for technique in techniques:
                    if technique in tokens:
                        found_techniques.append(technique)
                return ' '.join(found_techniques)

            def generate_recipe_name(ingredients, steps):
                techniques = extract_cooking_techniques(steps)
                ingredients_list = [lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b', str(ingredients).lower()) if word.isalpha() and word not in stop_words]
                main_ingredients = random.sample(ingredients_list, min(3, len(ingredients_list)))

                name_parts = set()
                if techniques:
                    name_parts.add(random.choice(techniques.split()))
                name_parts.update(main_ingredients)
                return " ".join(name_parts).title()

            def generate_recipe_name_v2(ingredients):
                ingredients_list = [lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b', str(ingredients).lower()) if word.isalpha() and word not in stop_words]
                random_adjective = random.choice(["Delicious", "Yummy", "Tasty", "Savory", "Mouthwatering"])
                main_ingredient = random.choice(ingredients_list) if ingredients_list else "Recipe"
                return f"{random_adjective} {main_ingredient.title()}"

            # Generate Recipe Names (Method 1)
            if st.button("Generate Recipe Names (Method 1)"):
                RAW_recipe['generated_name_m1'] = RAW_recipe.apply(lambda x: generate_recipe_name(x['ingredients'], x['steps']), axis=1)
                st.success("Generated names using Method 1!")
                st.write(RAW_recipe[['name', 'generated_name_m1', 'ingredients']].head(10))
            st.markdown(
    """
    La première méthode permet de générer un nom de recette en utilisant les techniques de cuisine et les ingrédients présents dans les étapes de chaque recette. 
    On extrait les techniques de cuisine (comme "cuire", "frire" ou "griller") à partir des instructions, puis sélectionne aléatoirement des ingrédients principaux tout en éliminant les verbes.
    
    Le nom est créé en combinant ces éléments et est affiché dans un tableau interactif, dont sont présentés ici quelques exemples.
    """
)


            # Generate Recipe Names (Method 2)
            if st.button("Generate Recipe Names (Method 2)"):
                RAW_recipe['generated_name_m2'] = RAW_recipe['ingredients'].apply(generate_recipe_name_v2)
                st.success("Generated names using Method 2!")
                st.write(RAW_recipe[['name', 'generated_name_m2', 'ingredients']].head(10))
            st.markdown(
                """
                Dans la seconde méthode, on utilise un dataset externe de recettes public (epirecipes sur Kaggle). Un preprocess est nécessaire afin de pouvoir 
                comparer les similarités d'ingrédients (seuil à 60% minimum) entre les recettes du dataset externe et celui utilisé dans le cadre de ce projet, RAW_recipes. 
                
                Lorsque la condition de 60% d'ingrédients similaires est réunie entre 2 recettes (1 recette de notre dataset et 1 recette du dataset externe), 
                on utilise le nom de la recette correspondante pour générer un nouveau nom.
                """
)

      