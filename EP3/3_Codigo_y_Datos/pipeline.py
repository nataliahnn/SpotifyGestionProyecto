# %% [markdown]
# # Pipeline de modelado: predicción de popularidad en Spotify
# Proyecto SCY1102-002D. Fases CRISP-DM 2-5: preparación, modelado, evaluación y auditoría de equidad.
# %%
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
SEED = 42
df = pd.read_csv("spotify.csv")
print("Filas/columnas originales:", df.shape)
# %% [markdown]
# ## 1. Preparación de datos (RF2)
# Deduplicación estricta por `track_id` para evitar fuga de datos entre train y test.
# %%
n0 = len(df)
dups = int(df.track_id.duplicated().sum())
df = df.drop_duplicates("track_id", keep="first").dropna(subset=["track_name","artists"]).reset_index(drop=True)
print(f"Duplicados por track_id eliminados: {dups} ({dups/n0:.1%}); filas finales: {len(df)}")
num = ["duration_ms","danceability","energy","key","loudness","mode","speechiness","acousticness",
       "instrumentalness","liveness","valence","tempo","time_signature"]
df["explicit"] = df.explicit.astype(int)
X = df[num+["explicit"]].copy()
X["genre"] = df.track_genre.astype("category").cat.codes   # codificación ordinal (árboles)
y = df.popularity
# %% [markdown]
# Partición ~80/20 estratificada por género y agrupada por (artista, título): una misma canción con
# distinto track_id (p. ej. reedición de álbum) nunca queda a la vez en train y test. En una prueba
# previa con partición no agrupada el MAE salía artificialmente menor (39% vs 32% de mejora).
# %%
from sklearn.model_selection import StratifiedGroupKFold
grupos = df.artists + "|" + df.track_name
idx_tr, idx_te = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, df.track_genre, grupos))
idx_tr, idx_te = df.index[idx_tr], df.index[idx_te]
Xtr, Xte, ytr, yte = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]
print("Train:", len(Xtr), "Test:", len(Xte))
# %% [markdown]
# ## 2. Modelado (RF3): baseline y modelos
# %%
def ev(name, pred):
    return {"modelo": name, "MAE": mean_absolute_error(yte, pred),
            "RMSE": mean_squared_error(yte, pred) ** 0.5}
res, preds = [], {}
preds["Baseline (media global)"] = np.full(len(yte), ytr.mean())
lin = Ridge(alpha=1.0).fit(Xtr, ytr);                      preds["Ridge"] = lin.predict(Xte)
rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
preds["Random Forest"] = rf.predict(Xte)
gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=SEED).fit(Xtr, ytr)
preds["Gradient Boosting"] = gb.predict(Xte)
cols_sg = [c for c in X.columns if c != "genre"]
rf_sg = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED).fit(Xtr[cols_sg], ytr)
preds["Random Forest (sin género)"] = rf_sg.predict(Xte[cols_sg])
for k, p in preds.items(): res.append(ev(k, p))
R = pd.DataFrame(res); base = R.MAE.iloc[0]
R["Mejora vs baseline"] = 1 - R.MAE / base
print(R.round(3).to_string(index=False))
# %% [markdown]
# ## 3. Evaluación y Go/No-Go (RNF1)
# Meta: MAE al menos 20% menor que el baseline.
# %%
best = R.iloc[1:][R.modelo.iloc[1:] != "Random Forest (sin género)"].sort_values("MAE").iloc[0]
print("Mejor modelo:", best.modelo, f"| mejora {best['Mejora vs baseline']:.1%}", "| Cumple RNF1:", best["Mejora vs baseline"] >= 0.20)
# %% [markdown]
# ## 4. Auditoría de equidad por género (RF4, RNF2)
# Regla: si MAE del género > 1,30 × MAE global, se congela la inferencia para ese género.
# %%
pb = preds[best.modelo]
a = pd.DataFrame({"genre": df.loc[idx_te, "track_genre"].values, "err": np.abs(yte.values - pb)})
g = a.groupby("genre").err.agg(["mean","count"]).rename(columns={"mean":"MAE"})
glob = a.err.mean(); g["ratio"] = g.MAE / glob
flag = g[g.ratio > 1.30].sort_values("ratio", ascending=False)
print(f"MAE global {glob:.2f}; géneros sobre umbral 1,30x: {len(flag)} de {len(g)}")
print(flag.round(2).head(10))
# %% [markdown]
# ## 5. Gráficos y exportación de resultados
# %%
fig, ax = plt.subplots(figsize=(7,3.6))
ax.barh(R.modelo[::-1], R.MAE[::-1], color=["#1DB954" if m==best.modelo else "#9AA5AD" for m in R.modelo[::-1]])
ax.set_xlabel("MAE (puntos de popularidad)"); ax.set_title("MAE en test por modelo")
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("fig_mae.png", dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(7,3.6))
ax.hist(g.ratio, bins=25, color="#1DB954"); ax.axvline(1.30, color="#C0392B", ls="--")
ax.set_xlabel("MAE del género / MAE global"); ax.set_ylabel("Cantidad de géneros"); ax.set_title("Auditoría de equidad por género (umbral 1,30)")
for s in ["top","right"]: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("fig_equidad.png", dpi=160); plt.close()
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
out = {"filas_originales": n0, "duplicados": dups, "filas_finales": len(df), "train": len(Xtr), "test": len(Xte),
       "resultados": R.round(4).to_dict("records"), "mejor": best.modelo, "mejora_mejor": float(best["Mejora vs baseline"]),
       "mae_global_mejor": float(glob), "generos": int(len(g)), "generos_sobre_umbral": int(len(flag)),
       "peores_generos": flag.round(2).head(5).reset_index().to_dict("records"),
       "importancias_rf": imp.round(3).head(6).to_dict()}
json.dump(out, open("resultados.json","w"), ensure_ascii=False, indent=1)
R.to_csv("metricas_modelos.csv", index=False); g.round(3).to_csv("auditoria_genero.csv")
print(imp.head(6))
