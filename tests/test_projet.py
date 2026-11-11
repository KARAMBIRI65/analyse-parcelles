import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model as M  # noqa: E402
from src.dvf import prix_m2_par_commune, terrains_a_batir  # noqa: E402
from src.simulation import generer_parcelles, surcouts_construction  # noqa: E402


def test_simulation_reproductible():
    a, b = generer_parcelles(3000, seed=1), generer_parcelles(3000, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["zone_plu"]) <= {"U", "AU", "A", "N"}
    assert (a.loc[a["zone_inondable"] == "rouge", "constructible"].mean()) < 0.1


def test_surcouts():
    d = pd.DataFrame({"zone_inondable": ["non", "bleue"], "alea_argiles": ["faible", "fort"],
                      "pente_pct": [3.0, 10.0], "viabilise": [True, False]})
    s = surcouts_construction(d)
    assert s.iloc[0] == 0
    assert s.iloc[1] == 15_000 + 14_000 + 5 * 1_800 + 20_000


def test_gain_attendu_et_decision():
    g = M.gain_attendu(1.0, 100_000, 10_000, 500, 80_000)
    assert g == 10_000
    assert M.gain_attendu(0.0, 100_000, 10_000, 1000, 80_000) == 1000 * M.PRIX_M2_AGRICOLE - 80_000
    d = M.decider([10_000, -5_000, -50_000], [80_000, 80_000, 80_000], marge=0)
    assert list(d) == ["Acheter", "Négocier", "Éviter"]


def test_pipeline_complet():
    df = generer_parcelles(1500, seed=3)
    clf = M.classifieur_logistique().fit(M.preparer(df), df["constructible"])
    cons = df[df["constructible"] == 1]
    reg = M.regresseur_xgb().fit(M.preparer(cons), cons["valeur_marche"])
    r = M.analyser(df.iloc[[0]], clf, reg, marge=5_000)
    assert r["decision"] in {"Acheter", "Négocier", "Éviter"}
    assert 0 <= r["probabilite_constructible"] <= 1


def test_dvf_terrains_a_batir():
    dvf = pd.DataFrame({
        "id_mutation": ["m1", "m1", "m2", "m3", "m4"] + [f"x{i}" for i in range(6)],
        "date_mutation": "2024-01-01",
        "nature_mutation": ["Vente"] * 4 + ["Echange"] + ["Vente"] * 6,
        "valeur_fonciere": [60_000, 60_000, 200_000, 50_000, 40_000] + [50_000] * 6,
        "code_commune": "33001", "nom_commune": "Test",
        "type_local": [np.nan, np.nan, "Maison", np.nan, np.nan] + [np.nan] * 6,
        "code_nature_culture": ["AB", "AB", "S", "AB", "AB"] + ["AB"] * 6,
        "surface_terrain": [300, 300, 500, 500, 400] + [500] * 6,
        "longitude": 0.0, "latitude": 45.0,
    })
    v = terrains_a_batir(dvf)
    assert "m2" not in set(v["id_mutation"])        # maison : exclue
    assert "m4" not in set(v["id_mutation"])        # échange : exclu
    m1 = v.loc[v["id_mutation"] == "m1"]
    if len(m1):                                     # m1 = 2 lignes -> surface sommée
        assert m1["surface_terrain"].iloc[0] == 600
    p = prix_m2_par_commune(v)
    assert len(p) == 1 and p["prix_m2_commune"].iloc[0] == 100
