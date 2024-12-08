import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cdist

# Configuration de la page
st.set_page_config(page_title="Analyse des Recettes", layout="wide")

# Fond d'écran
background_image_url = "https://media.cdnws.com/_i/96967/25307/3948/10/deco-table-avec-assiette-noire.jpeg"  # Remplacez par le chemin de votre image
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Titre de l'application
st.title("Interface d'Analyse des Données de Recettes")

# Sidebar: Upload CSV files
st.sidebar.header("Téléchargez vos fichiers CSV (RAW_recipes.csv et RAW_interactions.csv)")
uploaded_recipes_file = st.sidebar.file_uploader("Choisissez un fichier CSV pour les recettes", type=["csv"])
uploaded_interactions_file = st.sidebar.file_uploader("Choisissez un fichier CSV pour les interactions", type=["csv"])

# Charger les datasets
@st.cache_data
def load_data(file_1, file_2):
    """Charge les fichiers CSV et retourne deux DataFrames."""
    return pd.read_csv(file_1), pd.read_csv(file_2)

def preprocess_data(recipes_df, interactions_df):
    """Prépare les données (convertit les dates, calcule les moyennes, supprime les valeurs aberrantes)."""
    # Convertir les colonnes de dates en format datetime
    recipes_df['submitted'] = pd.to_datetime(recipes_df['submitted'])
    interactions_df['date'] = pd.to_datetime(interactions_df['date'])

    # Calcul de la note moyenne et du nombre d'interactions par recette
    recipe_ratings = interactions_df.groupby('recipe_id')['rating'].mean().reset_index()
    recipe_ratings.columns = ['id', 'average_rating']
    interaction_counts = interactions_df.groupby('recipe_id')['rating'].count().reset_index()
    interaction_counts.columns = ['id', 'n_interactions']

    # Fusionner avec le dataset des recettes
    recipes_df = recipes_df.merge(recipe_ratings, on='id', how='left')
    recipes_df = recipes_df.merge(interaction_counts, on='id', how='left')

    # Remplir les valeurs manquantes
    recipes_df['average_rating'].fillna(0, inplace=True)
    recipes_df['n_interactions'].fillna(0, inplace=True)

    # Éliminer les valeurs aberrantes
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    recipes_df = remove_outliers(recipes_df, 'minutes')
    recipes_df = remove_outliers(recipes_df, 'n_steps')
    recipes_df = remove_outliers(recipes_df, 'n_ingredients')

    return recipes_df

def display_overview(df, title):
    """Affiche un aperçu des données."""
    st.write(f"### Aperçu de {title}")
    st.write(df.head())

def display_statistics(df, title):
    """Affiche les statistiques de base des données."""
    st.write(f"### Statistiques de base de {title}")
    st.write(df.describe(include='all'))

def display_visualizations(df, title, key_prefix):
    """Affiche les visualisations des données."""
    st.write(f"### Visualisations de {title}")

    plot_type = st.sidebar.selectbox(f"Choisissez le type de graphique pour {title}", ["Scatter Plot", "Heatmap", "PCA", "Histogram", "Box Plot"], key=f"{key_prefix}_plot_type")

    if plot_type == "Scatter Plot":
        st.write("### Scatter Plot")
        numerical_columns = df.select_dtypes(include=[np.number, 'datetime']).columns
        x_axis = st.selectbox("X-axis", options=numerical_columns, key=f"{key_prefix}_x_axis")
        y_axis = st.selectbox("Y-axis", options=numerical_columns, key=f"{key_prefix}_y_axis")
        color_by = st.selectbox("Color by", options=numerical_columns.insert(0, None), key=f"{key_prefix}_color_by")

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=color_by, ax=ax)
        x_margin = (df[x_axis].max() - df[x_axis].min()) * 0.1
        y_margin = (df[y_axis].max() - df[y_axis].min()) * 0.1
        ax.set_xlim(df[x_axis].min() - x_margin, df[x_axis].max() + x_margin)
        ax.set_ylim(df[y_axis].min() - y_margin, df[y_axis].max() + y_margin)
        st.pyplot(fig)

    elif plot_type == "Heatmap":
        st.write("### Correlation Heatmap")
        correlation_matrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif plot_type == "PCA":
        st.write("### PCA Analysis")
        max_components = min(len(df.select_dtypes(include=[np.number]).columns), len(df))
        n_components = st.sidebar.slider("Number of Components", 2, max_components, 2, key=f"{key_prefix}_n_components")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
        st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
        fig, ax = plt.subplots()
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", ax=ax)
        x_margin = (pca_df["PC1"].max() - pca_df["PC1"].min()) * 0.1
        y_margin = (pca_df["PC2"].max() - pca_df["PC2"].min()) * 0.1
        ax.set_xlim(pca_df["PC1"].min() - x_margin, pca_df["PC1"].max() + x_margin)
        ax.set_ylim(pca_df["PC2"].min() - y_margin, pca_df["PC2"].max() + y_margin)
        st.pyplot(fig)

    elif plot_type == "Histogram":
        st.write("### Histogram")
        numerical_columns = df.select_dtypes(include=[np.number, 'datetime']).columns
        column = st.selectbox("Select Column", options=numerical_columns, key=f"{key_prefix}_hist_column")
        fig, ax = plt.subplots()
        sns.histplot(df[column], bins=30, kde=True, ax=ax)
        ax.set_title(f'Histogram of {column}')
        x_margin = (df[column].max() - df[column].min()) * 0.1
        ax.set_xlim(df[column].min() - x_margin, df[column].max() + x_margin)
        st.pyplot(fig)

    elif plot_type == "Box Plot":
        st.write("### Box Plot")
        numerical_columns = df.select_dtypes(include=[np.number, 'datetime']).columns
        column = st.selectbox("Select Column", options=numerical_columns, key=f"{key_prefix}_box_column")
        fig, ax = plt.subplots()
        sns.boxplot(data=df[column], ax=ax, showfliers=True)
        ax.set_title(f'Box Plot of {column}')
        y_margin = (df[column].max() - df[column].min()) * 0.1
        ax.set_ylim(df[column].min() - y_margin, df[column].max() + y_margin)
        st.pyplot(fig)

if uploaded_recipes_file and uploaded_interactions_file:
    if uploaded_recipes_file.name == "RAW_recipes.csv" and uploaded_interactions_file.name == "RAW_interactions.csv":
        recipes_df, interactions_df = load_data(uploaded_recipes_file, uploaded_interactions_file)
        recipes_df = preprocess_data(recipes_df, interactions_df)

        st.write("### Aperçu des Données")
        st.write("#### Données des Recettes")
        st.write(recipes_df.head())
        st.write("#### Données des Interactions")
        st.write(interactions_df.head())

        st.write("### Prétraitement des Données")
        st.write("#### Données après Prétraitement")
        st.write(recipes_df.describe())

        st.write("### Q1: Quelles sont les caractéristiques des recettes les plus populaires ?")
        popular_recipes = recipes_df[recipes_df['n_interactions'] > recipes_df['n_interactions'].quantile(0.75)]
        st.write("#### Recettes Populaires")
        st.write(popular_recipes.describe())

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        sns.histplot(popular_recipes['minutes'], bins=30, kde=True, ax=ax[0])
        ax[0].set_title('Distribution du Temps de Préparation')
        sns.histplot(popular_recipes['n_steps'], bins=30, kde=True, ax=ax[1])
        ax[1].set_title('Distribution du Nombre d\'Étapes')
        sns.histplot(popular_recipes['n_ingredients'], bins=30, kde=True, ax=ax[2])
        ax[2].set_title('Distribution du Nombre d\'Ingrédients')
        st.pyplot(fig)

        st.write("### Q2: Quels sont les facteurs influençant les notes des recettes ?")
        correlation_matrix = recipes_df[['minutes', 'n_steps', 'n_ingredients', 'average_rating']].corr()
        st.write("#### Matrice de Corrélation")
        st.write(correlation_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        sns.scatterplot(data=recipes_df, x='minutes', y='average_rating', ax=ax[0])
        ax[0].set_title('Note Moyenne vs. Temps de Préparation')
        sns.scatterplot(data=recipes_df, x='n_steps', y='average_rating', ax=ax[1])
        ax[1].set_title('Note Moyenne vs. Nombre d\'Étapes')
        sns.scatterplot(data=recipes_df, x='n_ingredients', y='average_rating', ax=ax[2])
        ax[2].set_title('Note Moyenne vs. Nombre d\'Ingrédients')
        st.pyplot(fig)

        st.write("### Q3: Visualisation des recettes par réduction de dimension pour déterminer les recettes qui se rapprochent de celles que l'on a déjà faites")
        recipe_features = recipes_df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(recipe_features)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        recipes_df['PCA1'] = pca_result[:, 0]
        recipes_df['PCA2'] = pca_result[:, 1]

        plt.figure(figsize=(10, 6))
        plt.scatter(recipes_df['PCA1'], recipes_df['PCA2'], c='blue', alpha=0.5, label='Recettes')
        plt.title('Visualisation des Recettes (PCA)')
        plt.xlabel('Composante 1')
        plt.ylabel('Composante 2')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        target_recipe = pca_result[0]
        distances = cdist([target_recipe], pca_result, metric='euclidean')
        recipes_df['Distance_to_Target'] = distances[0]
        closest_recipes = recipes_df.nsmallest(5, 'Distance_to_Target')
        st.write("#### Recettes les plus proches de la recette cible")
        st.write(closest_recipes)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=recipes_df, x='PCA1', y='PCA2', ax=ax, label='Recettes')
        sns.scatterplot(data=closest_recipes, x='PCA1', y='PCA2', ax=ax, color='red', label='Recettes Proches')
        plt.title('Recettes les plus proches de la recette cible')
        plt.xlabel('Composante 1')
        plt.ylabel('Composante 2')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

    else:
        st.write("Les fichiers téléchargés ne sont pas les bons. Analyse des fichiers téléchargés :")

        data1, data2 = load_data(uploaded_recipes_file, uploaded_interactions_file)

        for col in data1.columns:
            if pd.api.types.is_string_dtype(data1[col]):
                try:
                    data1[col] = pd.to_datetime(data1[col])
                except ValueError:
                    pass

        for col in data2.columns:
            if pd.api.types.is_string_dtype(data2[col]):
                try:
                    data2[col] = pd.to_datetime(data2[col])
                except ValueError:
                    pass

        display_overview(data1, uploaded_recipes_file.name)
        display_overview(data2, uploaded_interactions_file.name)
        display_statistics(data1, uploaded_recipes_file.name)
        display_statistics(data2, uploaded_interactions_file.name)
        display_visualizations(data1, uploaded_recipes_file.name, "file1")
        display_visualizations(data2, uploaded_interactions_file.name, "file2")

else:
    st.write("Veuillez télécharger les datasets requis (RAW_recipes.csv et RAW_interactions.csv) pour effectuer l'analyse.")