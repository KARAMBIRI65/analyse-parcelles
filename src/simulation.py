"""Génération d'un jeu de parcelles réaliste (données simulées).

Chaque ligne est une parcelle mise en vente dans un département fictif de 40 communes.
Les variables reprennent ce que l'on trouve dans les données ouvertes françaises :

- DVF (Demandes de valeurs foncières) : prix de vente, surface du terrain, commune ;
- PLU / Géoportail de l'urbanisme : zone U (urbaine), AU (à urbaniser), A ou N (agricole, naturelle) ;
- Géorisques : zone inondable (PPRI), retrait-gonflement des argiles ;
- RGE ALTI (IGN) : pente du terrain ;
- BPE (Insee) : distance aux écoles, aux commerces, à la gare.

Les ordres de grandeur (prix au m² des terrains à bâtir, surcoûts de construction)
s'inspirent de valeurs publiques, mais les parcelles sont inventées : aucune ne
correspond à un terrain réel. Le module `src/dvf.py` permet de recaler les prix
sur les vraies ventes DVF d'un département.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SURCOUT_INONDATION_BLEUE = 15_000   # plancher surélevé, matériaux adaptés (€)
SURCOUT_ARGILE = {"faible": 0, "moyen": 6_000, "fort": 14_000}  # étude G2 + fondations profondes
SURCOUT_PENTE_PAR_POINT = 1_800      # terrassement, murs de soutènement, par point de pente au-delà de 5 %
SURCOUT_VIABILISATION = 20_000       # raccordement eau, électricité, assainissement


def _communes(rng: np.random.Generator, n: int = 40) -> pd.DataFrame:
    """Communes : une ville-centre, des communes périurbaines et rurales."""
    dist = np.sort(rng.gamma(2.0, 9.0, n))
    dist[0] = 0.0
    # le prix des terrains baisse avec l'éloignement de la ville-centre
    prix_m2 = 260 * np.exp(-dist / 22) * rng.lognormal(0, 0.18, n) + 25
    return pd.DataFrame({
        "commune": [f"Commune_{i:02d}" for i in range(n)],
        "distance_centre_km": dist.round(1),
        "prix_m2_commune": prix_m2.round(0),
        "gare": rng.random(n) < np.clip(0.8 - dist / 50, 0.1, 0.9),
        "riviere": rng.random(n) < 0.45,
    })


def surcouts_construction(df: pd.DataFrame) -> pd.Series:
    """Surcoûts de construction (€) liés aux risques et à l'état du terrain."""
    s = np.where(df["zone_inondable"] == "bleue", SURCOUT_INONDATION_BLEUE, 0)
    s = s + df["alea_argiles"].map(SURCOUT_ARGILE).to_numpy()
    s = s + np.clip(df["pente_pct"] - 5, 0, None) * SURCOUT_PENTE_PAR_POINT
    s = s + np.where(df["viabilise"], 0, SURCOUT_VIABILISATION)
    return pd.Series(np.round(s, -2), index=df.index, name="surcouts")


def generer_parcelles(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    com = _communes(rng)
    idx = rng.choice(len(com), size=n, p=np.r_[0.12, np.full(len(com) - 1, 0.88 / (len(com) - 1))])
    df = com.iloc[idx].reset_index(drop=True)

    df.insert(0, "id_parcelle", [f"P{i:05d}" for i in range(n)])
    df["surface_m2"] = np.clip(rng.lognormal(np.log(650), 0.55, n), 150, 6000).round(0)

    # Zone PLU : plus on s'éloigne du centre, plus les terrains agricoles/naturels sont fréquents
    p_u = np.clip(0.70 - df["distance_centre_km"] / 70, 0.25, 0.75)
    p_au = np.full(n, 0.15)
    u = rng.random(n)
    df["zone_plu"] = np.where(u < p_u, "U", np.where(u < p_u + p_au, "AU", np.where(rng.random(n) < 0.6, "A", "N")))

    # Risque inondation : uniquement dans les communes traversées par une rivière
    df["distance_cours_eau_m"] = np.where(df["riviere"], rng.gamma(2.0, 350, n), rng.gamma(4.0, 900, n)).round(0)
    p_rouge = np.where(df["distance_cours_eau_m"] < 150, 0.55, np.where(df["distance_cours_eau_m"] < 400, 0.10, 0.0))
    p_bleue = np.where(df["distance_cours_eau_m"] < 150, 0.35, np.where(df["distance_cours_eau_m"] < 600, 0.30, 0.01))
    u = rng.random(n)
    df["zone_inondable"] = np.where(u < p_rouge, "rouge", np.where(u < p_rouge + p_bleue, "bleue", "non"))

    df["alea_argiles"] = rng.choice(["faible", "moyen", "fort"], n, p=[0.5, 0.33, 0.17])
    df["pente_pct"] = np.clip(rng.gamma(1.6, 3.5, n), 0, 45).round(1)
    df["viabilise"] = rng.random(n) < np.where(df["zone_plu"] == "U", 0.85, np.where(df["zone_plu"] == "AU", 0.35, 0.08))
    df["largeur_facade_m"] = np.clip(np.sqrt(df["surface_m2"]) * rng.lognormal(0, 0.35, n), 4, 80).round(1)
    df["distance_ecole_km"] = np.clip(rng.gamma(1.5, 0.8, n) + df["distance_centre_km"] / 30, 0.1, 15).round(2)
    df["distance_gare_km"] = np.where(df["gare"], rng.gamma(1.6, 1.2, n), 5 + rng.gamma(2, 4, n)).round(1)

    # Constructibilité « réelle » (permis accordé) : les règles du PLU et des PPRI,
    # plus des contraintes invisibles dans les données (servitudes, avis ABF, voisinage...)
    regle = (
        df["zone_plu"].isin(["U", "AU"])
        & (df["zone_inondable"] != "rouge")
        & (df["pente_pct"] < 25)
        & (df["largeur_facade_m"] >= 8)
    )
    p_ok = np.where(df["zone_plu"] == "U", 0.95, 0.70)
    p_ok = p_ok - 0.15 * (df["zone_inondable"] == "bleue") - 0.01 * df["pente_pct"]
    cache = rng.random(n) < np.clip(p_ok, 0.05, 1)
    exception = rng.random(n) < 0.02  # quelques terrains A/N constructibles (hameau, STECAL)
    df["constructible"] = ((regle & cache) | (~regle & exception)).astype(int)

    # Valeur de marché : un terrain constructible vaut des dizaines de fois un terrain agricole
    attrait = (
        np.exp(-df["distance_ecole_km"] / 6)
        * np.exp(-df["distance_gare_km"] / 40)
        * np.where(df["viabilise"], 1.12, 1.0)
        * np.where(df["zone_inondable"] == "bleue", 0.85, 1.0)
        * np.exp(-np.clip(df["pente_pct"] - 8, 0, None) / 40)
    )
    prix_m2 = df["prix_m2_commune"] * attrait * (df["surface_m2"] / 650) ** -0.30
    prix_m2_agricole = rng.uniform(0.6, 3.0, n)
    prix_m2 = np.where(df["constructible"] == 1, prix_m2, prix_m2_agricole)
    df["valeur_marche"] = (df["surface_m2"] * prix_m2 * rng.lognormal(0, 0.12, n)).round(-2).clip(lower=500)

    # Prix affiché par le vendeur : parfois surévalué, et certains vendeurs
    # affichent un prix de terrain à bâtir pour un terrain qui ne l'est pas (le piège classique).
    prix = df["valeur_marche"] * rng.lognormal(-0.06, 0.22, n)
    piege = (df["constructible"] == 0) & (rng.random(n) < 0.35)
    prix_si_constructible = df["surface_m2"] * df["prix_m2_commune"] * rng.uniform(0.4, 0.8, n)
    df["prix_demande"] = np.where(piege, prix_si_constructible, prix).round(-2).clip(min=500)

    df["surcouts"] = surcouts_construction(df)
    gain = df["valeur_marche"] - df["prix_demande"] - df["surcouts"]
    df["bonne_affaire"] = ((df["constructible"] == 1) & (gain > 0)).astype(int)

    return df.drop(columns=["gare", "riviere"])


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "data" / "parcelles.csv"
    d = generer_parcelles()
    d.to_csv(out, index=False)
    print(f"{len(d)} parcelles -> {out}")
    print(d[["constructible", "bonne_affaire"]].mean())
