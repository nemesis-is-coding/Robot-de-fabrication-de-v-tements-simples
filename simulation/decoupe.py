"""F3 : découpe simulée. Une lame (pointe) suit le contour de la pièce "corps"
du patron, posée sur un tissu plus grand. On mesure l'écart entre la
trajectoire réellement suivie et le contour théorique.
"""
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# On réutilise le générateur de patron (F1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "patron"))
from generer_patron import generer_patron

VITESSE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20   # m/s
SEUIL_MM = 2.0                                                 # critère F3
Z_LAME = 0.43

# --- Contour de la pièce "corps" (en mètres, centré sur la table) ---
corps = generer_patron(380, 420, 600, 30)[0]
pts = np.array(corps.contour + [corps.contour[0]]) / 1000.0
pts -= np.array([corps.largeur, corps.hauteur]) / 2000.0

XML = f"""
<mujoco>
  <option timestep="0.002" solver="CG" tolerance="1e-6"/>
  <size memory="50M"/>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="2 2 .1"/>
    <geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 0.38"
          friction="1 0.005 0.0001" rgba="0.6 0.45 0.3 1"/>
    <flexcomp type="grid" count="16 16 1" spacing="0.04 0.04 0.04"
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
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# La lame démarre sur le premier point du contour ; le tissu se pose d'abord
data.qpos[-2:] = pts[0]
data.ctrl[:] = pts[0]
while data.time < 2.0:
    mujoco.mj_step(model, data)

# Trajectoire de consigne : le long du contour, à vitesse constante
segments = np.diff(pts, axis=0)
longueurs = np.linalg.norm(segments, axis=1)
cumul = np.concatenate([[0], np.cumsum(longueurs)])
perimetre = cumul[-1]

def consigne(s):
    s = min(s, perimetre)
    i = min(np.searchsorted(cumul, s, side="right") - 1, len(segments) - 1)
    return pts[i] + segments[i] * (s - cumul[i]) / longueurs[i]

def distance_au_contour(p):
    d = np.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        d = min(d, np.linalg.norm(p - (a + t * ab)))
    return d

t_debut = data.time
ecarts = []
t0 = time.perf_counter()
while True:
    s = VITESSE * (data.time - t_debut)
    data.ctrl[:] = consigne(s)
    mujoco.mj_step(model, data)
    ecarts.append(distance_au_contour(data.qpos[-2:]))
    if s >= perimetre + 0.3 * VITESSE:   # petit délai pour finir le tour
        break
calcul = time.perf_counter() - t0

ecarts = np.array(ecarts) * 1000
print(f"Contour : {corps.largeur:.0f} x {corps.hauteur:.0f} mm, "
      f"périmètre {perimetre*1000:.0f} mm, vitesse {VITESSE*1000:.0f} mm/s")
print(f"Temps simulé de découpe : {data.time - t_debut:.1f} s "
      f"(calcul : {calcul:.1f} s)")
print(f"Écart au contour : moyen {ecarts.mean():.2f} mm, max {ecarts.max():.2f} mm")
print("F3 :", "VALIDÉ" if ecarts.max() <= SEUIL_MM else "NON VALIDÉ",
      f"(critère ±{SEUIL_MM:.0f} mm)")
