import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import zipfile
import os

# Streamlit App Title
st.title("Recipe Visualization and Similarity Finder")

# Sidebar Header
st.sidebar.header("Data Options")

# File Handling
uploaded_file = st.sidebar.file_uploader("Upload Recipe Dataset (ZIP file)", type="zip")

if uploaded_file is not None:
    # Extract the zip file
    output_directory = './Dataset_MTM'
    os.makedirs(output_directory, exist_ok=True)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    # Load the data
    data_path = os.path.join(output_directory, 'RAW_recipes.csv')
    recipes = pd.read_csv(data_path)

    # Display the first few rows
    st.subheader("Preview of Dataset")
    st.write(recipes.head())

    # Data Cleaning
    recipes_cleaned = recipes.copy()

    # Remove abnormal values
    abnormal_value = recipes_cleaned['minutes'].max()
    index_of_abnormal = recipes_cleaned[recipes_cleaned['minutes'] >= abnormal_value].index
    recipes_cleaned.drop(index_of_abnormal, inplace=True)

    # Sidebar Filters
    st.sidebar.subheader("Filter Recipes")
    max_minutes = st.sidebar.slider("Maximum Cooking Time (minutes)", 0, int(recipes_cleaned['minutes'].max()), 60)

    recipes_filtered = recipes_cleaned[recipes_cleaned['minutes'] <= max_minutes]

    # PCA Preparation
    st.subheader("PCA-based Recipe Visualization")
    selected_features = ['minutes', 'n_steps', 'tags', 'ingredients']
    recipes_filtered = recipes_filtered[selected_features].dropna()

    # Preprocessing for PCA
    recipes_filtered['tags_str'] = recipes_filtered['tags'].apply(lambda x: ' '.join(eval(x)))
    recipes_filtered['ingredients_str'] = recipes_filtered['ingredients'].apply(lambda x: ' '.join(eval(x)))

    vectorizer_tags = CountVectorizer()
    tags_vectorized = vectorizer_tags.fit_transform(recipes_filtered['tags_str'])

    vectorizer_ingredients = CountVectorizer()
    ingredients_vectorized = vectorizer_ingredients.fit_transform(recipes_filtered['ingredients_str'])

    combined_features = pd.concat([
        recipes_filtered[['minutes', 'n_steps']].reset_index(drop=True),
        pd.DataFrame(tags_vectorized.toarray()),
        pd.DataFrame(ingredients_vectorized.toarray())
    ], axis=1)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)

    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    recipes_filtered['pca1'] = pca_result[:, 0]
    recipes_filtered['pca2'] = pca_result[:, 1]

    # Plot PCA
    st.write("Interactive PCA Plot")
    fig, ax = plt.subplots()
    ax.scatter(recipes_filtered['pca1'], recipes_filtered['pca2'], alpha=0.5)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Recipe Clustering")
    st.pyplot(fig)

    # Find Similar Recipes
    st.subheader("Find Similar Recipes")
    selected_recipe_index = st.selectbox("Select a Recipe Index", recipes_filtered.index)
    selected_recipe = recipes_filtered.loc[selected_recipe_index]

    distances = np.sqrt(
        (recipes_filtered['pca1'] - selected_recipe['pca1'])**2 +
        (recipes_filtered['pca2'] - selected_recipe['pca2'])**2
    )
    recipes_filtered['distance'] = distances

    similar_recipes = recipes_filtered.sort_values(by='distance').head(10)
    st.write("Similar Recipes:")
    st.write(similar_recipes[['minutes', 'n_steps', 'tags_str', 'ingredients_str']])

else:
    st.info("Please upload the dataset to proceed.")
