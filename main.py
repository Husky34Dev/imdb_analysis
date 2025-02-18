import pandas as pd

# 1. Cargar el archivo TSV de name.basics
df = pd.read_csv(
    "Dataset/name.basics.tsv",
    sep="\t",
    low_memory=False,
    na_values="\\N",  # Trata "\N" como valores nulos
    keep_default_na=False  # Evita que pandas convierta "NA" a NaN
)

# 2. Filtrar solo filas donde 'knownForTitles' no sea NaN o vacío
df_filtered = df[df['knownForTitles'].notna()]

# 3. Separar las películas por coma en nuevas filas (un título por fila)
df_filtered = df_filtered.explode('knownForTitles')

# 4. Eliminar duplicados, si una persona tiene más de una fila para la misma película
df_filtered = df_filtered.drop_duplicates(subset=['nconst', 'knownForTitles'])

# 5. Ahora podemos eliminar personas que no están asociadas con títulos de películas de 'movies_clean'
# Cargar movies_clean para usarlo en la relación
movies_df = pd.read_csv("Dataset/clean/movies_clean.csv")

# Filtrar solo los 'nconst' que tienen títulos de películas en movies_clean
df_filtered = df_filtered[df_filtered['knownForTitles'].isin(movies_df['tconst'])]

# 6. Corregir 'birthYear' para asegurar que tenga el formato correcto (años de 4 dígitos)
df_filtered['birthYear'] = pd.to_numeric(df_filtered['birthYear'], errors='coerce')

# Corregir valores mal formateados (por ejemplo, valores demasiado grandes o pequeños)
df_filtered['birthYear'] = df_filtered['birthYear'].apply(lambda x: int(x) if 1900 <= x <= 2025 else None)

# 7. Guardar el DataFrame limpio en un archivo CSV
df_filtered.to_csv("actors_clean.csv", index=False, encoding="utf-8")

# Imprimir el resultado del proceso
print("¡Datos limpios de actores guardados en 'actors_clean.csv'!")
print(f"Filas originales: {len(df)} | Filas filtradas: {len(df_filtered)}")
