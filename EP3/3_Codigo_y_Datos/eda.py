"""Análisis exploratorio (EDA) de sesgos: GDPS-18.
Uso: python eda.py  (lee spotify.csv y escribe eda_resumen.csv y eda_generos.csv)"""
import pandas as pd
df = pd.read_csv("spotify.csv")
print("Filas originales:", len(df), "| track_id únicos:", df.track_id.nunique())
# 1. Popularidad por género (los 5 más altos y más bajos)
g = df.groupby("track_genre").popularity.agg(["mean", "median", "count"]).round(1)
print("\nPopularidad media por género: mayor", g["mean"].idxmax(), g["mean"].max(), "| menor", g["mean"].idxmin(), g["mean"].min())
# 2. Concentración por artista (artistas con más filas)
a = df.artists.value_counts()
print("\nArtista con más filas:", a.index[0], a.iloc[0], "| mediana de filas por artista:", a.median())
# 3. Contenido explícito, total y en Hip-Hop
exp_total = df.explicit.mean()
exp_hh = df[df.track_genre == "hip-hop"].explicit.mean() if (df.track_genre == "hip-hop").any() else float("nan")
print(f"\nExplicit total: {exp_total:.2%} | Hip-Hop: {exp_hh:.2%}")
# 4. Duplicados por track_id y por (artista, título)
print("Duplicados por track_id:", int(df.track_id.duplicated().sum()),
      "| duplicados por (artista, título):", int(df.duplicated(["artists", "track_name"]).sum()))
# Salidas
res = pd.DataFrame([
    ("filas_originales", len(df)), ("track_id_unicos", df.track_id.nunique()),
    ("duplicados_track_id", int(df.track_id.duplicated().sum())),
    ("duplicados_artista_titulo", int(df.duplicated(["artists", "track_name"]).sum())),
    ("explicit_total_pct", round(exp_total * 100, 2)), ("explicit_hiphop_pct", round(exp_hh * 100, 2)),
    ("popularidad_media_global", round(df.popularity.mean(), 2)),
    ("artista_mas_filas", f"{a.index[0]} ({a.iloc[0]})"), ("mediana_filas_por_artista", float(a.median())),
], columns=["indicador", "valor"])
res.to_csv("eda_resumen.csv", index=False)
g.sort_values("mean", ascending=False).to_csv("eda_generos.csv")
