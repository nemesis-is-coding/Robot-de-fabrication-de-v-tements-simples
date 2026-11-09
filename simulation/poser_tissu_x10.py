"""F2 : 10 poses avec variations aléatoires (position, hauteur de chute)."""
import sys
import mujoco
import numpy as np

chemin = sys.argv[1] if len(sys.argv) > 1 else "table.xml"
SEUIL_MM = 3.0   # critère de planéité à valider avec l'encadrant
dessus_table = 0.40
rng = np.random.default_rng(0)  # graine fixe : reproductible

model = mujoco.MjModel.from_xml_path(chemin)
resultats = []
for essai in range(10):
    data = mujoco.MjData(model)
    n = model.nflexvert
    dx, dy = rng.uniform(-0.03, 0.03, 2)
    dz = rng.uniform(0.0, 0.05)
    data.qpos[0:3 * n:3] += dx
    data.qpos[1:3 * n:3] += dy
    data.qpos[2:3 * n:3] += dz
    mujoco.mj_forward(model, data)
    while data.time < 3.0:
        mujoco.mj_step(model, data)
    h = data.flexvert_xpos[:, 2] - dessus_table
    ecart = (h.max() - h.min()) * 1000
    resultats.append(ecart)
    print(f"Essai {essai+1:2d} : décalage ({dx*1000:+.0f}, {dy*1000:+.0f}) mm, "
          f"chute +{dz*1000:.0f} mm -> planéité {ecart:.1f} mm")

ok = sum(r < SEUIL_MM for r in resultats)
print(f"\n{ok}/10 essais sous {SEUIL_MM} mm (max : {max(resultats):.1f} mm)")
