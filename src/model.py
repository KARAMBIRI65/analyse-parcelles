"""Modèles et règle de décision : acheter, négocier ou éviter une parcelle.

La décision est découpée en trois questions, chacune traitée séparément :

1. **Peut-on construire ?**  -> classifieur qui estime P(constructible)
2. **Combien vaut le terrain s'il est constructible ?**  -> régression du prix (log)
3. **Combien coûtera la construction en plus ?**  -> surcoûts calculés à partir des risques

Gain attendu = P × (valeur si constructible − surcoûts) + (1 − P) × valeur agricole − prix demandé
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

NUM = [
    "surface_m2", "distance_centre_km", "prix_m2_commune", "distance_cours_eau_m",
    "pente_pct", "largeur_facade_m", "distance_ecole_km", "distance_gare_km", "viabilise",
]
CAT = ["zone_plu", "zone_inondable", "alea_argiles"]
FEATURES = NUM + CAT

PRIX_M2_AGRICOLE = 1.5  # €/m², valeur d'un terrain non constructible (ordre de grandeur des terres agricoles)


def preprocesseur(scale: bool = True) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ], verbose_feature_names_out=False)


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["viabilise"] = X["viabilise"].astype(int)
    return X


# --- 1. Constructibilité -------------------------------------------------------

def classifieur_logistique() -> Pipeline:
    return Pipeline([("prep", preprocesseur()), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def classifieur_xgb(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("prep", preprocesseur(scale=False)),
        ("clf", XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.8, random_state=seed)),
    ])


# --- 2. Valeur si constructible ------------------------------------------------

class RegresseurLog:
    """Entraîne sur log(prix) puis repasse en euros (meilleur pour des prix très dispersés)."""

    def __init__(self, pipe: Pipeline):
        self.pipe = pipe

    def fit(self, X, y):
        self.pipe.fit(X, np.log(y))
        resid = np.log(y) - self.pipe.predict(X)
        self._corr = np.exp(resid.var() / 2)  # correction de biais du passage log -> euros
        return self

    def predict(self, X):
        return np.exp(self.pipe.predict(X)) * self._corr


def regresseur_ridge() -> RegresseurLog:
    return RegresseurLog(Pipeline([("prep", preprocesseur()), ("reg", Ridge(alpha=1.0))]))


def regresseur_xgb(seed: int = 0) -> RegresseurLog:
    return RegresseurLog(Pipeline([
        ("prep", preprocesseur(scale=False)),
        ("reg", XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04,
                             subsample=0.9, colsample_bytree=0.8, random_state=seed)),
    ]))


def mape(y, yhat) -> float:
    return float(np.mean(np.abs(yhat - y) / y))


# --- 3. Décision ---------------------------------------------------------------

def gain_attendu(p_constructible, valeur_constructible, surcouts, surface, prix_demande):
    valeur_agricole = surface * PRIX_M2_AGRICOLE
    return (p_constructible * (valeur_constructible - surcouts)
            + (1 - p_constructible) * valeur_agricole - prix_demande)


def prix_max_conseille(p_constructible, valeur_constructible, surcouts, surface, marge):
    """Prix au-delà duquel le gain attendu passe sous la marge visée."""
    return gain_attendu(p_constructible, valeur_constructible, surcouts, surface, 0) - marge


def decider(gain, prix_demande, marge, tolerance=0.10):
    """'Acheter' si gain >= marge ; 'Négocier' si une baisse de prix <= tolérance suffit ; sinon 'Éviter'."""
    gain = np.asarray(gain)
    prix_demande = np.asarray(prix_demande)
    return np.where(gain >= marge, "Acheter",
                    np.where(gain + tolerance * prix_demande >= marge, "Négocier", "Éviter"))


def gain_realise(df: pd.DataFrame, achete) -> np.ndarray:
    """Ce que rapporte vraiment l'achat (vérité terrain), parcelle par parcelle.

    Si le terrain est constructible : valeur de marché − prix − surcoûts.
    Sinon : on se retrouve avec un terrain agricole payé au prix demandé.
    """
    achete = np.asarray(achete, dtype=bool)
    g = np.where(df["constructible"] == 1,
                 df["valeur_marche"] - df["prix_demande"] - df["surcouts"],
                 df["valeur_marche"] - df["prix_demande"])
    return np.where(achete, g, 0.0)


def bilan_strategie(df: pd.DataFrame, achete, nom: str) -> dict:
    achete = np.asarray(achete, dtype=bool)
    g = gain_realise(df, achete)
    n = int(achete.sum())
    return {
        "strategie": nom,
        "parcelles_achetees": n,
        "dont_non_constructibles": int((achete & (df["constructible"].to_numpy() == 0)).sum()),
        "gain_total_k€": round(g.sum() / 1000, 0),
        "gain_moyen_par_achat_€": round(g.sum() / n, 0) if n else 0.0,
        "taux_bonnes_affaires": round(float(df["bonne_affaire"].to_numpy()[achete].mean()), 3) if n else 0.0,
    }


def choisir_marge(df_val: pd.DataFrame, gain_pred: np.ndarray, marges=None) -> tuple[float, pd.DataFrame]:
    """Choisit la marge minimale exigée qui maximise le gain réalisé sur la validation."""
    if marges is None:
        marges = np.arange(0, 40_001, 1_000)  # on exige au moins de ne pas perdre d'argent
    rows = [{"marge": m, "gain_total": gain_realise(df_val, gain_pred >= m).sum(),
             "n_achats": int((gain_pred >= m).sum())} for m in marges]
    t = pd.DataFrame(rows)
    return float(t.loc[t["gain_total"].idxmax(), "marge"]), t


# --- Fiche d'analyse d'une parcelle ---------------------------------------------

SEUIL_CONSTRUCTIBLE = 0.80  # on n'achète que si la constructibilité est très probable


def analyser(parcelle: dict | pd.DataFrame, clf, reg, marge: float) -> dict:
    """Recommandation complète pour une parcelle (dict ou DataFrame d'une ligne)."""
    from .simulation import surcouts_construction

    d = pd.DataFrame([parcelle]) if isinstance(parcelle, dict) else parcelle.reset_index(drop=True)
    X = preparer(d)
    p = float(clf.predict_proba(X)[0, 1])
    v = float(reg.predict(X)[0])
    s = float(surcouts_construction(d).iloc[0])
    prix = float(d["prix_demande"].iloc[0])
    surface = float(d["surface_m2"].iloc[0])
    g = float(gain_attendu(p, v, s, surface, prix))
    if p < SEUIL_CONSTRUCTIBLE:
        decision = "Éviter"
        raison = ("terrain très probablement non constructible" if p < 0.3
                  else f"constructibilité incertaine ({p:.0%}) : demander un certificat d'urbanisme")
    else:
        decision = str(decider(g, prix, marge))
        raison = {"Acheter": "prix inférieur à la valeur estimée, surcoûts compris",
                  "Négocier": "rentable avec une baisse de prix de 10 % au plus",
                  "Éviter": "trop cher par rapport à la valeur estimée et aux surcoûts"}[decision]
    return {
        "probabilite_constructible": round(p, 3),
        "valeur_estimee_si_constructible": round(v, -2),
        "surcouts_construction": round(s, -2),
        "prix_demande": prix,
        "gain_attendu": round(g, -2),
        "prix_max_conseille": max(0.0, round(float(prix_max_conseille(p, v, s, surface, marge)), -2)),
        "decision": decision,
        "raison": raison,
    }
