"""F2 : poser un tissu à plat sur une table et mesurer sa planéité."""
import sys
import mujoco
import numpy as np

chemin = sys.argv[1] if len(sys.argv) > 1 else "table.xml"
model = mujoco.MjModel.from_xml_path(chemin)
data = mujoco.MjData(model)

dessus_table = 0.40
duree = 3.0
while data.time < duree:
    mujoco.mj_step(model, data)

p = data.flexvert_xpos
hauteur = p[:, 2] - dessus_table
largeur = p[:, 0].max() - p[:, 0].min()
longueur = p[:, 1].max() - p[:, 1].min()

print(f"Dimensions du tissu : {largeur*1000:.0f} x {longueur*1000:.0f} mm")
print(f"Hauteur au-dessus de la table : min {hauteur.min()*1000:.1f} mm, "
      f"max {hauteur.max()*1000:.1f} mm")
print(f"Écart de planéité (max - min) : {(hauteur.max()-hauteur.min())*1000:.1f} mm")
print(f"Vitesse max des sommets : {np.abs(data.qvel).max():.4f} (m/s)")
print(f"Centre du tissu : x={p[:,0].mean()*1000:.0f} mm, y={p[:,1].mean()*1000:.0f} mm")
