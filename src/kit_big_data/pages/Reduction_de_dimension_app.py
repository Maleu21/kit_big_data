import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Set up the Streamlit app
st.title("Recipe Visualization and Similarity Finder")

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

# Sidebar for file upload
st.sidebar.header("Data Options")
uploaded_file = st.sidebar.file_uploader("Upload Recipe Dataset (CSV file)", type="csv")

if uploaded_file:
    try:
        # Load the dataset
        recipes = pd.read_csv(uploaded_file)
        st.write("Dataset loaded successfully!")
        st.write(recipes.head())  # Display the first few rows of the dataset

        # Preprocess and clean the data
        recipes_cleaned = recipes[['name', 'minutes', 'tags', 'n_steps', 'description', 'ingredients']].dropna()
        recipes_cleaned['tags'] = recipes_cleaned['tags'].apply(lambda x: ' '.join(eval(x)))
        recipes_cleaned['ingredients'] = recipes_cleaned['ingredients'].apply(lambda x: ' '.join(eval(x)))

        # Identify and remove anomalies in 'minutes' column
        q1 = recipes_cleaned['minutes'].quantile(0.25)
        q3 = recipes_cleaned['minutes'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        recipes_cleaned = recipes_cleaned[(recipes_cleaned['minutes'] >= lower_bound) & (recipes_cleaned['minutes'] <= upper_bound)]

        # Dimensionality reduction using PCA
        st.subheader("PCA Visualization")
        features = ['minutes', 'n_steps']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(recipes_cleaned[features])

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        recipes_cleaned['PCA1'] = pca_result[:, 0]
        recipes_cleaned['PCA2'] = pca_result[:, 1]

        # Visualize the PCA result
        fig, ax = plt.subplots()
        ax.scatter(recipes_cleaned['PCA1'], recipes_cleaned['PCA2'], alpha=0.5)
        ax.set_title("PCA Clustering of Recipes")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig)

        # Add histogram of 'minutes'
        st.subheader("Cooking Time Distribution")
        fig, ax = plt.subplots()
        ax.hist(recipes_cleaned['minutes'], bins=30, color='blue', alpha=0.7)
        ax.set_title("Distribution of Cooking Time")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Add a bar chart for most frequent tags
        st.subheader("Most Frequent Tags")
        all_tags = ' '.join(recipes_cleaned['tags']).split()
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        fig, ax = plt.subplots()
        tag_counts.plot(kind='bar', ax=ax, color='green', alpha=0.7)
        ax.set_title("Top 10 Most Frequent Tags")
        ax.set_xlabel("Tags")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Add interactive recipe similarity finder
        st.subheader("Find Similar Recipes")
        selected_recipe = st.selectbox("Select a Recipe to Find Similar Ones", recipes_cleaned['name'])
        selected_pca = recipes_cleaned[recipes_cleaned['name'] == selected_recipe][['PCA1', 'PCA2']].values[0]

        # Calculate distances to the selected recipe
        recipes_cleaned['distance'] = recipes_cleaned.apply(
            lambda row: ((row['PCA1'] - selected_pca[0])**2 + (row['PCA2'] - selected_pca[1])**2)**0.5, axis=1
        )
        similar_recipes = recipes_cleaned.sort_values(by='distance').head(10)
        st.write("Similar Recipes:")
        st.write(similar_recipes[['name', 'minutes', 'tags', 'ingredients']])

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload the dataset to proceed.")