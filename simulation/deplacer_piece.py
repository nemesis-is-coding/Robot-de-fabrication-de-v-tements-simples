"""F4 : le robot déplace une pièce de tissu entière avec une ventouse et la dépose
à une position cible.

  python deplacer_piece.py              # 1 essai, affiche le résultat
  python deplacer_piece.py --campagne 10  # 10 essais avec positions initiales aléatoires
  python deplacer_piece.py --campagne 10 --bruit 2   # vision moins précise (mm)
  python deplacer_piece.py --voir       # 1 essai dans la fenêtre MuJoCo
"""
import argparse
import time

import mujoco
import numpy as np

from robot_lib import NX, NY, SOMMETS_VENTOUSE, Z_TABLE, Robot

CIBLE = np.array([0.10, 0.08])   # position visée du centre de la pièce (m)
SEUIL_POS_MM = 3.0               # critère F4 : erreur de position
SEUIL_ANGLE_DEG = 1.0            # erreur d'orientation tolérée
Z_SAISIE, Z_PLACE, Z_LEVAGE = 0.410, 0.412, 0.70
BRUIT_VISION_MM = 1.0            # incertitude de la vision simulée (écart-type, mm)


def angle_piece(p):
    """Orientation de la pièce (degrés) : direction moyenne de ses côtés selon x."""
    v = np.mean([p[(NX - 1) * NY + iy, :2] - p[iy, :2] for iy in range(NY)], axis=0)
    return np.degrees(np.arctan2(v[1], v[0]))


def essai(rng, bruit_mm=BRUIT_VISION_MM, voir=False, temps_reel=False):
    r = Robot()
    n = r.m.nflexvert
    # Pièce posée avec un décalage aléatoire (le robot ne le connaît pas à l'avance)
    dx, dy = rng.uniform(-0.03, 0.03, 2)
    r.d.qpos[0:3 * n:3] += dx
    r.d.qpos[1:3 * n:3] += dy
    mujoco.mj_forward(r.m, r.d)

    def lancer(viewer=None):
        if viewer:
            r.sync = viewer.sync
        r.avancer(1.5, None, temps_reel)                        # le tissu se pose
        centre = r.sommets()[:, :2].mean(axis=0)                 # position réelle de la pièce
        centre = centre + rng.normal(0, bruit_mm / 1000, 2)      # ... vue avec une incertitude
        r.vers([*centre, 0.55], 2.0, temps_reel)                 # au-dessus de la pièce
        r.vers([*centre, Z_SAISIE], 1.5, temps_reel)             # la ventouse se pose
        r.saisir()
        r.avancer(0.3, None, temps_reel)
        r.vers([*centre, Z_LEVAGE], 2.0, temps_reel)             # soulève
        r.vers([*CIBLE, Z_LEVAGE], 2.5, temps_reel)              # transporte
        r.vers([*CIBLE, Z_PLACE], 2.0, temps_reel)               # repose
        r.relacher()
        r.avancer(0.5, None, temps_reel)
        r.vers([*CIBLE, 0.55], 1.5, temps_reel)                  # s'écarte
        r.avancer(1.5, None, temps_reel)                         # le tissu se stabilise
        return centre

    t0 = time.perf_counter()
    if voir:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(r.m, r.d) as v:
            centre = lancer(v)
            print("Terminé : ferme la fenêtre pour quitter.")
            while v.is_running():
                time.sleep(0.1)
    else:
        centre = lancer()

    p = r.sommets()
    h = (p[:, 2] - Z_TABLE) * 1000
    res = {
        "decalage_initial_mm": [round(dx * 1000), round(dy * 1000)],
        "erreur_position_mm": round(float(np.linalg.norm(p[:, :2].mean(axis=0) - CIBLE) * 1000), 2),
        "erreur_angle_deg": round(float(abs(angle_piece(p))), 2),
        "planeite_mm": round(float(h.max() - h.min()), 1),
        "stable": bool(np.isfinite(p).all()),
        "temps_calcul_s": round(time.perf_counter() - t0, 1),
    }
    res["ok"] = bool(res["stable"] and res["erreur_position_mm"] <= SEUIL_POS_MM
                     and res["erreur_angle_deg"] <= SEUIL_ANGLE_DEG)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campagne", type=int, default=0)
    ap.add_argument("--bruit", type=float, default=BRUIT_VISION_MM)
    ap.add_argument("--voir", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)           # graine fixe : reproductible

    if a.voir:
        res = essai(rng, a.bruit, voir=True, temps_reel=True)
        print(res)
        return
    resultats = []
    for i in range(a.campagne or 1):
        res = essai(rng, a.bruit)
        resultats.append(res)
        print(f"Essai {i+1:2d} : décalage {res['decalage_initial_mm']} mm -> "
              f"position {res['erreur_position_mm']} mm, angle {res['erreur_angle_deg']} deg, "
              f"planéité {res['planeite_mm']} mm -> {'OK' if res['ok'] else 'ECHEC'} "
              f"({res['temps_calcul_s']} s)", flush=True)
    ok = sum(r["ok"] for r in resultats)
    pos = [r["erreur_position_mm"] for r in resultats]
    print(f"\nRésultat : {ok}/{len(resultats)} essais conformes "
          f"(position <= {SEUIL_POS_MM} mm, angle <= {SEUIL_ANGLE_DEG} deg), "
          f"vision ±{a.bruit} mm")
    print(f"Erreur de position : moyenne {np.mean(pos):.2f} mm, max {np.max(pos):.2f} mm")


if __name__ == "__main__":
    main()
