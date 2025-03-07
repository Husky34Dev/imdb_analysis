import pandas as pd
import random

# Lista de géneros disponibles
available_genres = [
    "Action", "Comedy", "Drama", "Thriller", "Romance",
    "Sci-Fi", "Horror", "Fantasy", "Adventure", "Animation",
    "Family", "Crime", "Mystery"
]

# --- Usuarios Especializados ---
# Estos usuarios tienen gustos muy definidos para poder probar las recomendaciones.
special_users = [
    {
        "user_id": "user_superhero",
        "favorite_movies": "tt4154796, tt0848228",  # Avengers: Endgame, The Avengers
        "favorite_actors": "nm0000375, nm0262635",    # Robert Downey Jr., Chris Hemsworth
        "average_watch_time": 150,
        "preferred_genres": "Action, Adventure, Fantasy"  # Enfocado en películas de súper héroes
    },
    {
        "user_id": "user_drama",
        "favorite_movies": "tt0111161, tt0120338",  # The Shawshank Redemption, Titanic
        "favorite_actors": "nm0000151, nm0000209",    # Tom Hanks, Leonardo DiCaprio
        "average_watch_time": 120,
        "preferred_genres": "Drama, Romance"         # Enfocado en películas de drama y romance
    },
    {
        "user_id": "user_scifi",
        "favorite_movies": "tt0133093, tt1375666",  # The Matrix, Inception
        "favorite_actors": "nm0000206, nm0000138",    # Keanu Reeves, Laurence Fishburne
        "average_watch_time": 130,
        "preferred_genres": "Sci-Fi, Thriller"        # Enfocado en películas de ciencia ficción y thriller
    }
]


# --- Generar Usuarios Aleatorios ---
# Se crearán 97 usuarios adicionales con datos aleatorios para completar un total de 100.
all_users = special_users.copy()

for i in range(1, 98):  # Genera 97 usuarios (1 a 97)
    user_id = f"user_random_{i}"

    # Generar de 3 a 6 IDs de películas ficticias (formato: "tt" seguido de 7 dígitos)
    num_movies = random.randint(3, 6)
    fav_movies = []
    for _ in range(num_movies):
        movie_id = "tt" + str(random.randint(1000000, 9999999))
        fav_movies.append(movie_id)
    favorite_movies = ", ".join(fav_movies)

    # Generar de 3 a 6 IDs de actores ficticios (formato: "nm" seguido de 7 dígitos)
    num_actors = random.randint(3, 6)
    fav_actors = []
    for _ in range(num_actors):
        actor_id = "nm" + str(random.randint(1000000, 9999999))
        fav_actors.append(actor_id)
    favorite_actors = ", ".join(fav_actors)

    # Tiempo promedio de visualización aleatorio entre 60 y 180 minutos
    average_watch_time = random.randint(60, 180)

    # Seleccionar aleatoriamente entre 2 y 4 géneros
    num_genres = random.randint(2, 4)
    preferred_genres = ", ".join(random.sample(available_genres, num_genres))

    user = {
        "user_id": user_id,
        "favorite_movies": favorite_movies,
        "favorite_actors": favorite_actors,
        "average_watch_time": average_watch_time,
        "preferred_genres": preferred_genres
    }
    all_users.append(user)

# Crear DataFrame a partir de la lista de diccionarios
df_custom_users = pd.DataFrame(all_users)

# Guardar el DataFrame en un archivo CSV llamado "custom_users.csv"
df_custom_users.to_csv("custom_users.csv", index=False)

print("Archivo 'custom_users.csv' generado con 100 usuarios.")
