# =============================================================================
# DEMOSTRACIÓN AUTOMÁTICA DE RECOMENDACIONES PARA USUARIOS ESPECIALES
#
# Este script carga los datasets de películas, actores, ratings y usuarios
# (custom_users.csv). Luego, sin interacción manual, se generan y muestran las
# recomendaciones para usuarios con gustos específicos:
# "user_superhero", "user_drama" y "user_scifi".
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
    _ = pd.read_csv('Dataset/clean/actors_clean.csv')  # Cargamos actores aunque no se usan directamente
    ratings = pd.read_csv('Dataset/clean/ratings_clean.csv')

    # --- Filtrar Títulos en Español ---
    titles_akas = pd.read_csv('Dataset/clean/title_akas_clean.csv')
    # Selecciona registros donde la región es "ES" y conserva el de menor 'ordering' en caso de duplicados
    titles_akas_es = titles_akas[titles_akas['region'] == 'ES']
    titles_akas_es = titles_akas_es.sort_values('ordering').drop_duplicates('titleId', keep='first')

    # Realiza el merge entre películas y títulos en español
    movies = movies.merge(titles_akas_es[['titleId', 'title']],
                          left_on='tconst', right_on='titleId', how='inner')
    movies['primaryTitle'] = movies['title']
    movies.drop(['titleId', 'title'], axis=1, inplace=True)

    # --- Integrar Información de Ratings ---
    movies = movies.merge(ratings[['tconst', 'averageRating', 'numVotes']],
                          on='tconst', how='left')

    # --- Preprocesamiento de Datos ---
    # Convertir runtimeMinutes a numérico y rellenar valores nulos con la mediana
    movies['runtimeMinutes'] = pd.to_numeric(movies['runtimeMinutes'], errors='coerce')
    movies['runtimeMinutes'] = movies['runtimeMinutes'].fillna(movies['runtimeMinutes'].median())

    # Eliminar filas sin información de géneros y transformar la cadena de géneros en una lista
    movies = movies.dropna(subset=['genres'])
    movies['genres_list'] = movies['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')])

    # --- Crear Representación de Géneros y Entrenar el Modelo de Vecinos ---
    mlb = MultiLabelBinarizer()
    genre_features = mlb.fit_transform(movies['genres_list'])

    # Se utiliza NearestNeighbors con distancia coseno y algoritmo "brute" para búsquedas
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_model.fit(genre_features)

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
    Genera recomendaciones de películas combinando dos enfoques:
      1. Recomendaciones 'Tailored': Basadas en los géneros preferidos del usuario.
      2. Recomendaciones 'Diversificadas': Películas que NO contienen ninguno de los géneros preferidos.

    Parámetros:
      - user_id         : ID del usuario.
      - df_users        : DataFrame con la información de los usuarios.
      - movies          : DataFrame de películas procesadas.
      - mlb             : MultiLabelBinarizer para transformar géneros.
      - nn_model        : Modelo NearestNeighbors entrenado.
      - n_recommendations: Número total de recomendaciones a retornar.
      - diversified_ratio: Proporción de recomendaciones diversificadas (valor entre 0 y 1).
      - min_rating      : Rating mínimo requerido para considerar la película.

    Retorna:
      DataFrame con las columnas: tconst, primaryTitle, genres, runtimeMinutes, averageRating y total_score.
    """
    # --- Obtener Datos del Usuario ---
    user_row = df_users[df_users['user_id'] == user_id].iloc[0]
    preferred_genres = [g.strip() for g in user_row['preferred_genres'].split(',')]
    user_watch_time = user_row['average_watch_time']
    favorite_movies = [m.strip() for m in user_row['favorite_movies'].split(',')]

    # --- Recomendaciones Tailored (Por Géneros Preferidos) ---
    # Convertir géneros preferidos en vector binario
    user_vector = mlb.transform([preferred_genres])
    # Buscar los 200 vecinos más cercanos en base a la similitud de géneros
    distances, indices = nn_model.kneighbors(user_vector, n_neighbors=200)

    # Seleccionar las películas cercanas y calcular la similitud (1 - distancia)
    tailored_df = movies.iloc[indices[0]].copy()
    tailored_df['similarity'] = 1 - distances[0]
    # Excluir las películas favoritas del usuario
    tailored_df = tailored_df[~tailored_df['tconst'].isin(favorite_movies)]

    # Asegurar que el rating sea numérico y calcular puntajes:
    # - runtime_score: Penaliza la diferencia entre la duración de la película y el tiempo de visionado del usuario.
    # - rating_score : Normaliza el averageRating (de 0 a 1).
    # - total_score  : Multiplica la similitud, el rating y el runtime_score.
    tailored_df['averageRating'] = pd.to_numeric(tailored_df['averageRating'], errors='coerce').fillna(0)
    tailored_df['runtime_score'] = 1 / (1 + abs(tailored_df['runtimeMinutes'] - user_watch_time))
    tailored_df['rating_score'] = tailored_df['averageRating'] / 10.0
    tailored_df['total_score'] = tailored_df['similarity'] * tailored_df['rating_score'] * tailored_df['runtime_score']

    # Filtrar por rating mínimo y ordenar por total_score
    tailored_df = tailored_df[tailored_df['averageRating'] >= min_rating].sort_values('total_score', ascending=False)

    # Calcular cuántas recomendaciones serán de cada tipo
    num_diverse = int(n_recommendations * diversified_ratio)
    num_tailored = n_recommendations - num_diverse
    tailored_recs = tailored_df.head(num_tailored)

    # --- Recomendaciones Diversificadas (Sin Géneros Preferidos) ---
    # Seleccionar películas que NO contengan ninguno de los géneros preferidos del usuario
    diverse_df = movies[
        movies['genres_list'].apply(lambda genres: all(g not in preferred_genres for g in genres))].copy()
    diverse_df = diverse_df[~diverse_df['tconst'].isin(favorite_movies)]
    diverse_df['averageRating'] = pd.to_numeric(diverse_df['averageRating'], errors='coerce').fillna(0)
    diverse_df['runtime_score'] = 1 / (1 + abs(diverse_df['runtimeMinutes'] - user_watch_time))
    diverse_df['rating_score'] = diverse_df['averageRating'] / 10.0
    # En este caso, no se utiliza 'similarity'
    diverse_df['total_score'] = diverse_df['rating_score'] * diverse_df['runtime_score']
    diverse_df = diverse_df[diverse_df['averageRating'] >= min_rating].sort_values('total_score', ascending=False)
    diverse_recs = diverse_df.head(num_diverse)

    # --- Combinar y Ordenar las Recomendaciones Finales ---
    final_recs = pd.concat([tailored_recs, diverse_recs]).drop_duplicates(subset='tconst')
    final_recs = final_recs.sort_values('total_score', ascending=False)

    return final_recs[['tconst', 'primaryTitle', 'genres', 'runtimeMinutes', 'averageRating', 'total_score']]


# =============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# =============================================================================

if __name__ == '__main__':
    # --- Cargar y Preprocesar Datos ---
    movies, mlb, nn_model = load_and_process_movies()
    df_users = load_custom_users()

    # --- Definir Usuarios Especiales a Evaluar ---
    special_user_ids = ["user_superhero", "user_drama", "user_scifi"]

    # --- Iterar Sobre Cada Usuario y Mostrar Recomendaciones ---
    for user_id in special_user_ids:
        print("==========================================")
        print(f"Recomendaciones para el usuario: {user_id}\n")

        # Mostrar datos del usuario
        print("Datos del usuario:")
        print(df_users[df_users['user_id'] == user_id].to_string(index=False))
        print("\nPelículas recomendadas:")

        # Generar y mostrar las recomendaciones (puedes ajustar los parámetros según necesites)
        recs = recommend_movies_for_user(user_id, df_users, movies, mlb, nn_model,
                                         n_recommendations=10, diversified_ratio=0.5, min_rating=7.0)
        print(recs.to_string(index=False))
        print("==========================================\n")
