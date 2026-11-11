"""Analyse d'une parcelle en ligne de commande.

Exemple :
    python analyser_parcelle.py --surface 800 --prix 95000 --zone-plu U --inondable non \
        --argiles moyen --pente 6 --distance-centre 12 --prix-m2-commune 150
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import model as M

ap = argparse.ArgumentParser(description="Faut-il acheter cette parcelle pour construire ?")
ap.add_argument("--surface", type=float, required=True, help="surface du terrain (m²)")
ap.add_argument("--prix", type=float, required=True, help="prix demandé (€)")
ap.add_argument("--zone-plu", choices=["U", "AU", "A", "N"], default="U")
ap.add_argument("--inondable", choices=["non", "bleue", "rouge"], default="non")
ap.add_argument("--argiles", choices=["faible", "moyen", "fort"], default="faible")
ap.add_argument("--pente", type=float, default=3.0, help="pente moyenne (%%)")
ap.add_argument("--non-viabilise", action="store_true")
ap.add_argument("--facade", type=float, default=18.0, help="largeur de façade sur rue (m)")
ap.add_argument("--distance-centre", type=float, default=10.0, help="distance à la ville-centre (km)")
ap.add_argument("--prix-m2-commune", type=float, default=150.0, help="prix médian des terrains de la commune (€/m²)")
ap.add_argument("--distance-cours-eau", type=float, default=2000.0)
ap.add_argument("--distance-ecole", type=float, default=1.0)
ap.add_argument("--distance-gare", type=float, default=3.0)
a = ap.parse_args()

df = pd.read_csv(Path(__file__).parent / "data" / "parcelles.csv")
clf = M.classifieur_logistique().fit(M.preparer(df), df["constructible"])
cons = df[df["constructible"] == 1]
reg = M.regresseur_xgb().fit(M.preparer(cons), cons["valeur_marche"])
marge = json.loads((Path(__file__).parent / "results" / "parametres.json").read_text())["marge"]

parcelle = {
    "surface_m2": a.surface, "prix_demande": a.prix, "zone_plu": a.zone_plu,
    "zone_inondable": a.inondable, "alea_argiles": a.argiles, "pente_pct": a.pente,
    "viabilise": not a.non_viabilise, "largeur_facade_m": a.facade,
    "distance_centre_km": a.distance_centre, "prix_m2_commune": a.prix_m2_commune,
    "distance_cours_eau_m": a.distance_cours_eau, "distance_ecole_km": a.distance_ecole,
    "distance_gare_km": a.distance_gare,
}
r = M.analyser(parcelle, clf, reg, marge)


def eur(x):
    return f"{x:>10,.0f} €".replace(",", " ")


print(f"\n  Décision : {r['decision'].upper()}  ({r['raison']})\n")
print(f"  Probabilité d'être constructible : {r['probabilite_constructible']:>10.0%}")
print(f"  Valeur estimée si constructible  : {eur(r['valeur_estimee_si_constructible'])}")
print(f"  Surcoûts de construction         : {eur(r['surcouts_construction'])}")
print(f"  Prix demandé                     : {eur(r['prix_demande'])}")
print(f"  Gain attendu                     : {eur(r['gain_attendu'])}")
print(f"  Prix maximum conseillé           : {eur(r['prix_max_conseille'])}\n")
