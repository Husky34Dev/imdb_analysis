# =============================================================================
# DEMOSTRACIÓN AUTOMÁTICA DE RECOMENDACIONES PARA USUARIOS ESPECIALES
#
# Este script carga los datasets de películas, actores, ratings y usuarios
# (custom_users.csv). Luego, sin interacción manual, se generan y muestran las
# recomendaciones para usuarios con gustos específicos:
# "user_superhero", "user_drama" y "user_scifi".
#
# MODIFICADO PARA CREAR un archivo CSV 'recommendations.csv'
# =============================================================================

# ------------------------- Importar Librerías -------------------------
import pandas as pd  # Para manipulación de datos en DataFrames
import numpy as np  # Para operaciones numéricas
from sklearn.preprocessing import MultiLabelBinarizer  # Para transformar listas en vectores binarios
from sklearn.neighbors import NearestNeighbors  # Para encontrar vecinos similares (métrica coseno)

# --------------------- Configuraciones de Visualización ---------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# =============================================================================
# FUNCIONES DE CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================

def load_and_process_movies():
    """
    Carga y procesa los datasets relacionados con las películas:
      - Carga películas, actores, ratings y títulos alternativos.
      - Filtra títulos en español (región "ES") y actualiza el título principal.
      - Integra la información de ratings.
      - Preprocesa la duración y los géneros.
      - Crea la representación binaria de los géneros y entrena el modelo de vecinos.

    Retorna:
      movies  : DataFrame de películas procesadas.
      mlb     : Instancia de MultiLabelBinarizer ajustada a los géneros.
      nn_model: Modelo NearestNeighbors entrenado sobre la representación de géneros.
    """
    # --- Cargar Datasets de Películas, Actores y Ratings ---
    movies = pd.read_csv('Dataset/clean/movies_clean.csv')
    _ = pd.read_csv('Dataset/clean/actors_clean.csv')  # Actores no se usa por el momento
    ratings = pd.read_csv('Dataset/clean/ratings_clean.csv')

    # --- Filtrar Títulos en Español ---
    titles_akas = pd.read_csv('Dataset/clean/title_akas_clean.csv')
    titles_akas_es = titles_akas[
        (titles_akas['region'] == 'ES')
    ].sort_values('ordering').drop_duplicates('titleId', keep='first')

    # --- Unir Datos ---

    movies = movies.merge(
        titles_akas_es[['titleId', 'title']],
        left_on='tconst',
        right_on='titleId',
        how='inner'
    )
    movies['primaryTitle'] = movies['title']
    movies.drop(['titleId', 'title'], axis=1, inplace=True)

    # --- Integrar Ratings ---
    movies = movies.merge(
        ratings[['tconst', 'averageRating', 'numVotes']],
        on='tconst',
        how='left'
    )

    # --- Preprocesamiento de Datos ---
    movies['runtimeMinutes'] = pd.to_numeric(movies['runtimeMinutes'], errors='coerce').fillna(
        movies['runtimeMinutes'].median()).astype(int)

    # Nos aseguramos de que no haya películas sin género
    movies = movies.dropna(subset=['genres'])

    # Convertir géneros a lista y excluir "Documentary" y "Music"
    movies['genres_list'] = movies['genres'].apply(
        lambda x: [genre.strip() for genre in x.split(',')]
    )
    movies = movies[
        ~movies['genres_list'].apply(
            lambda x: any(g in x for g in ['Documentary', 'Music'])
        )
    ]

    # --- Entrenar Modelo (basado en géneros) ---
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(movies['genres_list'])
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute').fit(genre_features)

    return movies, mlb, nn_model


def load_custom_users():
    """
    Carga el CSV de usuarios personalizados.

    Retorna:
      DataFrame con la información de los usuarios.
    """
    return pd.read_csv("Dataset/clean/custom_users.csv")


# =============================================================================
# FUNCIÓN DE RECOMENDACIÓN
# =============================================================================

def recommend_movies_for_user(user_id, df_users, movies, mlb, nn_model,
                              n_recommendations=10, diversified_ratio=0.5, min_rating=7.0):
    """
    Genera recomendaciones con fórmula de scoring (70% similitud, 30% rating + runtime).
    Retorna un DataFrame con las películas recomendadas para un usuario concreto.
    """
    # --- Obtener Datos del Usuario ---
    user_row = df_users[df_users['user_id'] == user_id].iloc[0]
    preferred_genres = [g.strip() for g in user_row['preferred_genres'].split(',')]
    user_watch_time = user_row['average_watch_time']
    favorite_movies = [m.strip() for m in user_row['favorite_movies'].split(',')]

    # --- Recomendaciones de géneros preferidos (Tailored) ---
    user_vector = mlb.transform([preferred_genres])
    distances, indices = nn_model.kneighbors(user_vector, n_neighbors=200)

    tailored_df = movies.iloc[indices[0]].copy()
    tailored_df['similarity'] = 1 - distances[0]
    tailored_df = tailored_df[~tailored_df['tconst'].isin(favorite_movies)]

    # Cálculo de scores
    tailored_df['averageRating'] = pd.to_numeric(tailored_df['averageRating'], errors='coerce').fillna(0)
    tailored_df['runtime_score'] = 1 / (1 + abs(tailored_df['runtimeMinutes'] - user_watch_time))
    tailored_df['rating_score'] = tailored_df['averageRating'] / 10.0

    tailored_df['total_score'] = (
            0.7 * tailored_df['similarity'] +
            0.3 * ((tailored_df['rating_score'] + tailored_df['runtime_score']) / 2)
    ).round(2)

    # Filtrar las que no cumplan el min_rating
    tailored_df = tailored_df[tailored_df['averageRating'] >= min_rating].sort_values('total_score', ascending=False)

    # --- Recomendaciones Diversificadas ---
    diverse_df = movies[
        movies['genres_list'].apply(lambda genres: all(g not in preferred_genres for g in genres))
    ].copy()
    diverse_df = diverse_df[~diverse_df['tconst'].isin(favorite_movies)]

    # Similaridad = 0 para géneros no preferidos
    diverse_df['similarity'] = 0.0
    diverse_df['averageRating'] = pd.to_numeric(diverse_df['averageRating'], errors='coerce').fillna(0)
    diverse_df['runtime_score'] = 1 / (1 + abs(diverse_df['runtimeMinutes'] - user_watch_time))
    diverse_df['rating_score'] = diverse_df['averageRating'] / 10.0

    diverse_df['total_score'] = (
            0.7 * diverse_df['similarity'] +
            0.3 * ((diverse_df['rating_score'] + diverse_df['runtime_score']) / 2)
    ).round(2)

    diverse_df = diverse_df[diverse_df['averageRating'] >= min_rating].sort_values('total_score', ascending=False)

    # --- Combinar Resultados (Tailored + Diversificado) ---
    num_diverse = int(n_recommendations * diversified_ratio)
    num_tailored = n_recommendations - num_diverse

    final_recs = pd.concat([
        tailored_df.head(num_tailored),
        diverse_df.head(num_diverse)
    ]).drop_duplicates(subset='tconst')

    # --- Añadir la info del usuario para saber a quién va dirigida la recomendación ---
    final_recs['user_id'] = user_id

    # Reordenar columnas para claridad
    final_recs = final_recs[[
        'user_id', 'tconst', 'primaryTitle', 'genres',
        'runtimeMinutes', 'averageRating', 'total_score'
    ]]

    return final_recs


# =============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# =============================================================================

if __name__ == '__main__':
    # --- Cargar y Preprocesar Datos ---
    movies, mlb, nn_model = load_and_process_movies()
    df_users = load_custom_users()

    # --- Definir Usuarios Especiales a Evaluar ---
    special_user_ids = ["user_superhero", "user_drama", "user_scifi"]

    # Lista para ir guardando las recomendaciones de cada usuario
    all_recommendations = []

    # --- Iterar Sobre Cada Usuario y Mostrar Recomendaciones ---
    for user_id in special_user_ids:
        print("==========================================")
        print(f"Recomendaciones para el usuario: {user_id}\n")

        # Mostrar datos del usuario
        print("Datos del usuario:")
        print(df_users[df_users['user_id'] == user_id].to_string(index=False))
        print("\nPelículas recomendadas:")

        # Generar las recomendaciones
        recs = recommend_movies_for_user(
            user_id,
            df_users,
            movies,
            mlb,
            nn_model,
            n_recommendations=10,
            diversified_ratio=0.5,
            min_rating=8.0 ##Puedes alterar esto para peliculas con mucha nota
        )
        print(recs.to_string(index=False))

        # Guardar en la lista
        all_recommendations.append(recs)

        print("==========================================\n")

    # =========================================================================
    # DF FINAL PARA EXPORTACIÓN A CSV
    # =========================================================================

    # Unir todas las recomendaciones en un solo DataFrame
    df_recommendations = pd.concat(all_recommendations, ignore_index=True)

    # Guardar el DataFrame final en un archivo CSV llamado "recommendations.csv"
    df_recommendations.to_csv('recommendations.csv', index=False)
    print("Archivo 'recommendations.csv' generado correctamente.")

    # IMPORTANTE para Power BI:
    # Mostramos un mensaje final
    print("***** DataFrame FINAL PARA POWER BI *****")
    print(df_recommendations.head(20))
