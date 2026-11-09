"""Niveau 1 : patron (F1) -> tissu sur table (F2) -> découpe simulée (F3).

Usage :
  python niveau1.py                     # un sac complet (4 pièces)
  python niveau1.py --campagne 10       # validation : 10 découpes consécutives
  python niveau1.py --vitesse 0.1       # vitesse de la lame en m/s
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src" / "patron"))
from generer_patron import generer_patron

SEUIL_MM = 2.0        # critère F3 : écart au contour
Z_LAME = 0.43
PAS = 0.04            # maille du tissu (m)
MARGE_TISSU = 0.10    # tissu plus grand que la pièce, de chaque côté (m)


def scene_xml(largeur_mm, hauteur_mm):
    nx = int(np.ceil((largeur_mm / 1000 + 2 * MARGE_TISSU) / PAS)) + 1
    ny = int(np.ceil((hauteur_mm / 1000 + 2 * MARGE_TISSU) / PAS)) + 1
    return f"""
<mujoco>
  <option timestep="0.002" solver="CG" tolerance="1e-6"/>
  <size memory="50M"/>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="2 2 .1"/>
    <geom name="table" type="box" size="0.6 0.5 0.02" pos="0 0 0.38"
          friction="1 0.005 0.0001" rgba="0.6 0.45 0.3 1"/>
    <flexcomp type="grid" count="{nx} {ny} 1" spacing="{PAS} {PAS} {PAS}"
              pos="0 0 0.42" radius="0.002" name="cloth" dim="2" mass="0.2">
      <contact selfcollide="none" friction="1 0.005 0.0001"/>
      <edge equality="true"/>
    </flexcomp>
    <body name="lame" pos="0 0 {Z_LAME}">
      <joint name="lx" type="slide" axis="1 0 0"/>
      <joint name="ly" type="slide" axis="0 1 0"/>
      <geom type="sphere" size="0.004" mass="1" rgba="1 0 0 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="lx" kp="20000" kv="250"/>
    <position joint="ly" kp="20000" kv="250"/>
  </actuator>
</mujoco>
"""


def distance_au_contour(p, pts):
    d = np.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        d = min(d, np.linalg.norm(p - (a + t * ab)))
    return d


def decouper(piece, vitesse, rng):
    """Pose le tissu (position aléatoire), puis découpe le contour de la pièce."""
    model = mujoco.MjModel.from_xml_string(scene_xml(piece.largeur, piece.hauteur))
    data = mujoco.MjData(model)
    n = model.nflexvert

    # F2 : tissu posé avec un décalage et une hauteur de chute aléatoires
    dx, dy = rng.uniform(-0.03, 0.03, 2)
    dz = rng.uniform(0.0, 0.03)
    data.qpos[0:3 * n:3] += dx
    data.qpos[1:3 * n:3] += dy
    data.qpos[2:3 * n:3] += dz
    mujoco.mj_forward(model, data)
    while data.time < 1.5:
        mujoco.mj_step(model, data)
    haut = data.flexvert_xpos[:, 2] - 0.40
    planeite = (haut.max() - haut.min()) * 1000
    centre = data.flexvert_xpos[:, :2].mean(axis=0)

    # F3 : contour centré sur le tissu
    pts = np.array(piece.contour + [piece.contour[0]]) / 1000.0
    pts = pts - np.array([piece.largeur, piece.hauteur]) / 2000.0 + centre
    segments = np.diff(pts, axis=0)
    longueurs = np.linalg.norm(segments, axis=1)
    cumul = np.concatenate([[0], np.cumsum(longueurs)])
    perimetre = cumul[-1]

    def consigne(s):
        s = min(s, perimetre)
        i = min(np.searchsorted(cumul, s, side="right") - 1, len(segments) - 1)
        return pts[i] + segments[i] * (s - cumul[i]) / longueurs[i]

    data.qpos[-2:] = pts[0]
    data.ctrl[:] = pts[0]
    mujoco.mj_forward(model, data)

    t_debut, t0 = data.time, time.perf_counter()
    ecarts, trace, k = [], [], 0
    while True:
        s = vitesse * (data.time - t_debut)
        data.ctrl[:] = consigne(s)
        mujoco.mj_step(model, data)
        p = data.qpos[-2:].copy()
        ecarts.append(distance_au_contour(p, pts))
        if k % 5 == 0:
            trace.append(p)
        k += 1
        if s >= perimetre + 0.3 * vitesse:
            break
    trace = np.array(trace)
    etendue = (trace.max(axis=0) - trace.min(axis=0)) * 1000
    erreur_dim = max(abs(etendue[0] - piece.largeur), abs(etendue[1] - piece.hauteur))
    ecarts = np.array(ecarts) * 1000
    return {
        "piece": piece.nom,
        "decalage_mm": [round(dx * 1000), round(dy * 1000)],
        "planeite_mm": round(planeite, 2),
        "ecart_max_mm": round(float(ecarts.max()), 2),
        "ecart_moyen_mm": round(float(ecarts.mean()), 3),
        "erreur_dimension_mm": round(float(erreur_dim), 2),
        "temps_simule_s": round(data.time - t_debut, 1),
        "temps_calcul_s": round(time.perf_counter() - t0, 1),
        "ok": bool(ecarts.max() <= SEUIL_MM and erreur_dim <= SEUIL_MM),
        "_contour": pts, "_trace": trace,
    }


def tracer(resultats, chemin):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib absent : pas d'image)")
        return
    n = len(resultats)
    cols = min(n, 4)
    lignes = int(np.ceil(n / cols))
    fig, axes = plt.subplots(lignes, cols, figsize=(3.5 * cols, 3.5 * lignes),
                             squeeze=False)
    for ax, r in zip(axes.flat, resultats):
        c, t = r["_contour"], r["_trace"]
        ax.plot(c[:, 0], c[:, 1], "k-", lw=1, label="contour du patron")
        ax.plot(t[:, 0], t[:, 1], "r.", ms=2, label="trajectoire de la lame")
        ax.set_aspect("equal")
        ax.set_title(f"{r['piece']} : écart max {r['ecart_max_mm']} mm", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    print(f"Image : {chemin}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--largeur", type=float, default=380)
    ap.add_argument("--hauteur", type=float, default=420)
    ap.add_argument("--anse-longueur", type=float, default=600)
    ap.add_argument("--anse-largeur", type=float, default=30)
    ap.add_argument("--vitesse", type=float, default=0.2)
    ap.add_argument("--campagne", type=int, default=0)
    a = ap.parse_args()

    t_total = time.perf_counter()
    pieces = generer_patron(a.largeur, a.hauteur, a.anse_longueur, a.anse_largeur)
    print("F1 - patron :", ", ".join(
        f"{p.nom} x{p.quantite} ({p.largeur:.0f}x{p.hauteur:.0f} mm)" for p in pieces))

    rng = np.random.default_rng(0)   # graine fixe : résultats reproductibles
    if a.campagne:
        a_couper = [pieces[0]] * a.campagne
    else:
        a_couper = [p for p in pieces for _ in range(p.quantite)]

    resultats = []
    for i, p in enumerate(a_couper, 1):
        r = decouper(p, a.vitesse, rng)
        resultats.append(r)
        print(f"Découpe {i:2d}/{len(a_couper)} {r['piece']:5s} : décalage "
              f"{r['decalage_mm']} mm, planéité {r['planeite_mm']} mm, "
              f"écart max {r['ecart_max_mm']} mm, dimension {r['erreur_dimension_mm']} mm "
              f"-> {'OK' if r['ok'] else 'ECHEC'}", flush=True)

    ok = sum(r["ok"] for r in resultats)
    print(f"\nRésultat : {ok}/{len(resultats)} découpes conformes "
          f"(critère ±{SEUIL_MM:.0f} mm), vitesse {a.vitesse * 1000:.0f} mm/s")
    print("NIVEAU 1 :", "VALIDÉ" if ok == len(resultats) else "NON VALIDÉ")

    sortie = Path("sortie")
    sortie.mkdir(exist_ok=True)
    nom = "campagne" if a.campagne else "sac"
    tracer(resultats, sortie / f"niveau1_{nom}.png")
    with open(sortie / f"niveau1_{nom}.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in resultats], f, indent=2)
    print(f"Rapport : {sortie / f'niveau1_{nom}.json'}  "
          f"(durée totale {time.perf_counter() - t_total:.0f} s)")


if __name__ == "__main__":
    main()
