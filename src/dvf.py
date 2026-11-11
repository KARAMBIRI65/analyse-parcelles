"""Brancher le projet sur les vraies ventes de terrains (DVF géolocalisées, data.gouv.fr).

DVF recense toutes les ventes immobilières en France (hors Alsace-Moselle et Mayotte)
depuis 2014. Ce module télécharge le fichier d'un département, garde les ventes de
**terrains à bâtir** et calcule un prix médian au m² par commune. Ce prix peut ensuite
remplacer la colonne `prix_m2_commune` du jeu simulé.

Exemple :
    python -m src.dvf --dep 33 --annee 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"
COLONNES = [
    "id_mutation", "date_mutation", "nature_mutation", "valeur_fonciere",
    "code_commune", "nom_commune", "type_local", "code_nature_culture",
    "surface_terrain", "longitude", "latitude",
]
DATA = Path(__file__).resolve().parents[1] / "data"


def charger_dvf(dep: str, annee: int, cache: Path = DATA) -> pd.DataFrame:
    """Lit le fichier DVF du département (téléchargé une seule fois puis mis en cache)."""
    fichier = cache / f"dvf_{dep}_{annee}.csv.gz"
    if not fichier.exists():
        url = URL.format(annee=annee, dep=dep)
        print(f"Téléchargement de {url} ...")
        pd.read_csv(url, usecols=COLONNES, dtype={"code_commune": str}, low_memory=False) \
          .to_csv(fichier, index=False, compression="gzip")
    return pd.read_csv(fichier, dtype={"code_commune": str}, low_memory=False)


def terrains_a_batir(dvf: pd.DataFrame) -> pd.DataFrame:
    """Ventes simples de terrains à bâtir (code de nature de culture 'AB'), sans bâtiment.

    Une mutation DVF peut contenir plusieurs lignes (plusieurs parcelles ou locaux).
    On ne garde que les mutations composées uniquement de terrains à bâtir, puis on
    somme les surfaces : le prix (valeur_fonciere) est celui de la mutation entière.
    """
    d = dvf[dvf["nature_mutation"] == "Vente"].copy()
    lignes_ab = d.groupby("id_mutation")["code_nature_culture"].apply(lambda s: (s == "AB").all())
    sans_bati = d.groupby("id_mutation")["type_local"].apply(lambda s: s.isna().all())
    garder = lignes_ab[lignes_ab & sans_bati.reindex(lignes_ab.index)].index
    d = d[d["id_mutation"].isin(garder)]
    m = d.groupby("id_mutation").agg(
        date_mutation=("date_mutation", "first"),
        valeur_fonciere=("valeur_fonciere", "first"),
        surface_terrain=("surface_terrain", "sum"),
        code_commune=("code_commune", "first"),
        nom_commune=("nom_commune", "first"),
        longitude=("longitude", "mean"),
        latitude=("latitude", "mean"),
    ).reset_index()
    m = m[(m["surface_terrain"] >= 100) & (m["valeur_fonciere"] >= 1000)]
    m["prix_m2"] = m["valeur_fonciere"] / m["surface_terrain"]
    # on retire les valeurs aberrantes (erreurs de saisie, ventes entre proches...)
    bas, haut = m["prix_m2"].quantile([0.02, 0.98])
    return m[m["prix_m2"].between(bas, haut)]


def prix_m2_par_commune(ventes: pd.DataFrame, min_ventes: int = 5) -> pd.DataFrame:
    """Prix médian au m² des terrains à bâtir, par commune (communes avec assez de ventes)."""
    p = ventes.groupby(["code_commune", "nom_commune"]).agg(
        prix_m2_commune=("prix_m2", "median"), nb_ventes=("prix_m2", "size")).reset_index()
    return p[p["nb_ventes"] >= min_ventes].sort_values("prix_m2_commune", ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prix des terrains à bâtir par commune (DVF)")
    ap.add_argument("--dep", default="33", help="numéro du département, ex. 33, 69, 2A")
    ap.add_argument("--annee", type=int, default=2024)
    a = ap.parse_args()
    ventes = terrains_a_batir(charger_dvf(a.dep, a.annee))
    prix = prix_m2_par_commune(ventes)
    out = DATA / f"prix_terrains_{a.dep}_{a.annee}.csv"
    prix.to_csv(out, index=False)
    print(f"{len(ventes)} ventes de terrains à bâtir, {len(prix)} communes -> {out}")
    print(prix.head(10).to_string(index=False))
