"""Niveau 2 (premier test, F4) : un robot à 4 axes saisit un coin du tissu.

Contient : la scène MuJoCo (table, tissu, robot), la cinématique inverse et la séquence.

  python robot_saisie.py            # calcul seul, affiche le résultat
  python robot_saisie.py --voir     # ouvre la fenêtre MuJoCo et joue la scène
  python robot_saisie.py --planche  # enregistre planche.png (5 images de la séquence)
"""
import argparse
import time

import mujoco
import numpy as np


# Géométrie du bras (m)
BASE = (0.0, -0.65)       # position (x, y) de l'axe du robot
Z_EPAULE = 0.42           # hauteur de l'épaule
L1, L2, LP = 0.45, 0.42, 0.14   # bras, avant-bras, longueur poignet+pince
PAS = 0.04                # maille du tissu


def scene_xml(nx=11, ny=12, masse=0.1):
    return f"""
<mujoco model="robot_habillement">
  <compiler angle="radian"/>
  <option timestep="0.002" solver="CG" tolerance="1e-6" integrator="implicitfast"/>
  <size memory="50M"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.75 0.85 0.95" rgb2="0.95 0.95 0.95" width="256" height="256"/>
    <texture name="sol" type="2d" builtin="checker" rgb1="0.82 0.82 0.82" rgb2="0.75 0.75 0.75" width="128" height="128"/>
    <material name="sol" texture="sol" texrepeat="6 6" reflectance="0.1"/>
  </asset>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light pos="0 -1 2.5" dir="0 0.3 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="sol" type="plane" size="3 3 .1" material="sol" contype="1" conaffinity="1"/>
    <!-- table : dessus à z = 0.40 -->
    <geom name="table" type="box" size="0.6 0.5 0.02" pos="0 0 0.38"
          friction="1 0.005 0.0001" rgba="0.62 0.46 0.3 1" contype="1" conaffinity="1"/>
    <geom type="box" size="0.03 0.03 0.18" pos="0.55 0.45 0.18" rgba="0.4 0.3 0.2 1"/>
    <geom type="box" size="0.03 0.03 0.18" pos="-0.55 0.45 0.18" rgba="0.4 0.3 0.2 1"/>
    <geom type="box" size="0.03 0.03 0.18" pos="0.55 -0.45 0.18" rgba="0.4 0.3 0.2 1"/>
    <geom type="box" size="0.03 0.03 0.18" pos="-0.55 -0.45 0.18" rgba="0.4 0.3 0.2 1"/>

    <!-- tissu : pièce 0.40 x 0.44 m -->
    <flexcomp type="grid" count="{nx} {ny} 1" spacing="{PAS} {PAS} {PAS}"
              pos="0 0 0.42" radius="0.002" name="tissu" dim="2" mass="{masse}"
              rgba="0.25 0.5 0.85 1">
      <contact selfcollide="none" friction="1 0.005 0.0001"/>
      <edge equality="true"/>
    </flexcomp>

    <!-- robot -->
    <geom type="cylinder" fromto="{BASE[0]} {BASE[1]} 0 {BASE[0]} {BASE[1]} 0.38" size="0.09" rgba="0.3 0.3 0.35 1"/>
    <body name="base" pos="{BASE[0]} {BASE[1]} {Z_EPAULE-0.04}" gravcomp="1">
      <joint damping="2" armature="0.05" name="lacet" type="hinge" axis="0 0 1" range="-2.5 2.5"/>
      <geom type="cylinder" fromto="0 0 0 0 0 0.04" size="0.08" rgba="0.9 0.5 0.1 1"/>
      <body name="bras" pos="0 0 0.04" gravcomp="1">
        <joint damping="2" armature="0.05" name="epaule" type="hinge" axis="1 0 0" range="-1.6 1.6"/>
        <geom type="sphere" size="0.06" rgba="0.2 0.2 0.25 1"/>
        <geom type="capsule" fromto="0 0 0 0 {L1} 0" size="0.04" rgba="0.9 0.5 0.1 1"/>
        <body name="avantbras" pos="0 {L1} 0" gravcomp="1">
          <joint damping="2" armature="0.05" name="coude" type="hinge" axis="1 0 0" range="-2.6 2.6"/>
          <geom type="sphere" size="0.05" rgba="0.2 0.2 0.25 1"/>
          <geom type="capsule" fromto="0 0 0 0 {L2} 0" size="0.032" rgba="0.9 0.5 0.1 1"/>
          <body name="poignet" pos="0 {L2} 0" gravcomp="1">
            <joint damping="2" armature="0.05" name="poignet" type="hinge" axis="1 0 0" range="-2.6 2.6"/>
            <geom type="sphere" size="0.035" rgba="0.2 0.2 0.25 1"/>
            <body name="pince" pos="0 0 0" gravcomp="1">
              <geom type="box" size="0.03 0.02 0.012" pos="0 0.04 0" rgba="0.3 0.3 0.35 1"/>
              <geom type="capsule" fromto="0 0 0 0 {LP-0.03} 0" size="0.012" rgba="0.3 0.3 0.35 1"/>
              <site name="pointe" pos="0 {LP} 0" size="0.004"/>
              <body name="doigt_g" pos="-0.02 {LP-0.04} 0" gravcomp="1">
                <joint damping="2" armature="0.05" name="doigt_g" type="slide" axis="1 0 0" range="0 0.02"/>
                <geom type="box" size="0.004 0.04 0.012" pos="0 0.02 0" rgba="0.15 0.15 0.15 1"/>
              </body>
              <body name="doigt_d" pos="0.02 {LP-0.04} 0" gravcomp="1">
                <joint damping="2" armature="0.05" name="doigt_d" type="slide" axis="-1 0 0" range="0 0.02"/>
                <geom type="box" size="0.004 0.04 0.012" pos="0 0.02 0" rgba="0.15 0.15 0.15 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_lacet" joint="lacet" kp="300" kv="30"/>
    <position name="a_epaule" joint="epaule" kp="600" kv="60"/>
    <position name="a_coude" joint="coude" kp="400" kv="40"/>
    <position name="a_poignet" joint="poignet" kp="200" kv="20"/>
    <position name="a_doigt_g" joint="doigt_g" kp="500" kv="20"/>
    <position name="a_doigt_d" joint="doigt_d" kp="500" kv="20"/>
  </actuator>
  <equality>
    <connect name="saisie" body1="pince" body2="tissu_0" anchor="0 {LP} 0" active="false"/>
  </equality>
</mujoco>
"""


JOINTS = ["lacet", "epaule", "coude", "poignet"]


class IK:
    """Cinématique inverse numérique : position de la pointe + pince verticale."""

    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pointe")
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        self.qadr = [model.jnt_qposadr[j] for j in jid]
        self.vadr = [model.jnt_dofadr[j] for j in jid]
        self.lo = np.array([model.jnt_range[j][0] for j in jid])
        self.hi = np.array([model.jnt_range[j][1] for j in jid])

    def resoudre(self, cible, q0, iterations=200):
        d, m = self.d, self.m
        q = np.array(q0, dtype=float)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(iterations):
            for a, v in zip(self.qadr, q):
                d.qpos[a] = v
            mujoco.mj_forward(m, d)
            pos = d.site_xpos[self.site]
            axe = d.site_xmat[self.site].reshape(3, 3)[:, 1]   # axe de la pince (y local)
            e_pos = cible - pos
            e_rot = np.cross(axe, np.array([0.0, 0.0, -1.0]))   # pince vers le bas
            if np.linalg.norm(e_pos) < 1e-5 and np.linalg.norm(e_rot) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.vadr], jacr[:, self.vadr]])
            e = np.concatenate([e_pos, e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + 0.5 * dq, self.lo, self.hi)
        erreur = np.linalg.norm(cible - pos)
        return q, erreur


Z_TABLE = 0.40


def min_jerk(t):
    """Interpolation douce de 0 à 1."""
    t = np.clip(t, 0, 1)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class Scene:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(scene_xml())
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.ik = IK(m)
        self.act = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_" + n)
                    for n in ["lacet", "epaule", "coude", "poignet", "doigt_g", "doigt_d"]}
        self.eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "saisie")
        self.body_pince = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pince")
        self.body_coin = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tissu_0")
        self.q = np.array([0.0, 0.6, -1.2, -0.9])       # posture de départ
        d = self.d
        d.qpos[self.ik.qadr] = self.q                   # le robot démarre dans cette posture
        d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'doigt_g')]] = 0.02
        d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'doigt_d')]] = 0.02
        self.commander(self.q, 0.02)
        self.sync = None
        self.suivi_max = 0.0

    def commander(self, q, ouverture):
        for n, v in zip(["lacet", "epaule", "coude", "poignet"], q):
            self.d.ctrl[self.act[n]] = v
        self.d.ctrl[self.act["doigt_g"]] = ouverture
        self.d.ctrl[self.act["doigt_d"]] = ouverture

    def avancer(self, duree, q_fin=None, ouverture=0.02, temps_reel=False):
        """Avance de `duree` secondes en interpolant vers la posture q_fin."""
        q_deb = self.q.copy()
        t0 = self.d.time
        pas = self.m.opt.timestep
        n = int(duree / pas)
        for i in range(n):
            if q_fin is not None:
                self.q = q_deb + (q_fin - q_deb) * min_jerk((i + 1) / n)
            self.commander(self.q, ouverture)
            tic = time.perf_counter()
            mujoco.mj_step(self.m, self.d)
            if self.sync and i % 10 == 0:
                self.sync()
                if temps_reel:
                    reste = 10 * pas - (time.perf_counter() - tic)
                    if reste > 0:
                        time.sleep(reste)
        # erreur de suivi articulaire
        err = np.abs(self.d.qpos[self.ik.qadr] - self.q).max()
        self.suivi_max = max(self.suivi_max, err)

    def vers(self, cible, duree, ouverture=0.02, temps_reel=False):
        q, err = self.ik.resoudre(np.array(cible), self.q)
        if err > 2e-3:
            raise RuntimeError(f"cible hors de portée ({err*1000:.0f} mm)")
        self.avancer(duree, q, ouverture, temps_reel)

    def saisir(self):
        """Fixe le coin du tissu à la pince (saisie virtuelle) et ferme les doigts."""
        d, m = self.d, self.m
        mujoco.mj_forward(m, d)
        coin = d.xpos[self.body_coin]
        pince = d.xpos[self.body_pince]
        R = d.xmat[self.body_pince].reshape(3, 3)
        m.eq_data[self.eq, 0:3] = R.T @ (coin - pince)   # ancre dans le repère de la pince
        m.eq_data[self.eq, 3:6] = 0.0                    # sur le sommet du tissu
        d.eq_active[self.eq] = 1

    def relacher(self):
        self.d.eq_active[self.eq] = 0

    def sequence(self, temps_reel=False):
        d = self.d
        coin0 = d.flexvert_xpos[0].copy()
        r = {}
        tr = temps_reel
        mujoco.mj_forward(self.m, d)
        self.avancer(1.5, None, temps_reel=tr)                  # le tissu se pose
        pos_depart = d.flexvert_xpos.copy()
        self.vers(coin0 + [0, 0, 0.15], 2.0, temps_reel=tr)     # au-dessus du coin
        self.vers(coin0 + [0, 0, 0.012], 1.5, temps_reel=tr)    # descente
        err_approche = np.linalg.norm(d.site_xpos[self.ik.site] - (coin0 + [0, 0, 0.012]))
        self.saisir()
        self.avancer(0.6, None, ouverture=0.0, temps_reel=tr)   # doigts fermés
        self.vers(coin0 + [0, 0, 0.20], 2.0, ouverture=0.0, temps_reel=tr)   # soulève
        r["hauteur_coin_mm"] = (d.flexvert_xpos[0, 2] - Z_TABLE) * 1000
        r["hauteur_voisin_mm"] = (d.flexvert_xpos[1, 2] - Z_TABLE) * 1000
        pli = np.array([0.10, 0.12, 0.20])                      # vers le centre
        self.vers(coin0 + pli, 2.0, ouverture=0.0, temps_reel=tr)             # déplace
        self.vers(coin0 + pli * [1, 1, 0] + [0, 0, 0.02], 2.0, ouverture=0.0, temps_reel=tr)  # repose
        self.relacher()
        self.avancer(0.5, None, ouverture=0.02, temps_reel=tr)
        self.vers(coin0 + pli + [0, 0, 0.15], 1.5, temps_reel=tr)   # recule
        self.avancer(1.5, None, temps_reel=tr)                  # le tissu se stabilise
        p = d.flexvert_xpos
        r["erreur_approche_mm"] = err_approche * 1000
        r["deplacement_coin_mm"] = np.linalg.norm(p[0, :2] - pos_depart[0, :2]) * 1000
        r["hauteur_finale_max_mm"] = (p[:, 2].max() - Z_TABLE) * 1000
        r["sur_table"] = bool((np.abs(p[:, 0]) < 0.6).all() and (np.abs(p[:, 1]) < 0.5).all())
        r["stable"] = bool(np.isfinite(p).all())
        r["suivi_max_rad"] = self.suivi_max
        r["temps_simule_s"] = d.time
        return r


def planche(chemin="planche.png"):
    """Enregistre 5 images de la séquence (rendu hors écran, peut échouer selon le GPU/WSL)."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image
    s = Scene()
    rend = mujoco.Renderer(s.m, 360, 480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, -0.2, 0.5]
    cam.distance, cam.azimuth, cam.elevation = 1.9, 140, -22
    instants, images = [0.5, 4.8, 8.6, 11.0, 14.5], []

    def capter():
        if instants and s.d.time >= instants[0]:
            instants.pop(0)
            rend.update_scene(s.d, cam)
            images.append(rend.render().copy())
    s.sync = capter
    s.sequence()
    Image.fromarray(np.hstack(images)).save(chemin)
    print(f"Image enregistrée : {chemin}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voir", action="store_true", help="ouvrir la fenêtre MuJoCo")
    ap.add_argument("--planche", action="store_true", help="enregistrer planche.png")
    a = ap.parse_args()
    if a.planche:
        planche()
        return
    s = Scene()
    t0 = time.perf_counter()
    if a.voir:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(s.m, s.d) as v:
            s.sync = v.sync
            r = s.sequence(temps_reel=True)
            print("Terminé : ferme la fenêtre pour quitter.")
            while v.is_running():
                time.sleep(0.1)
    else:
        r = s.sequence()
    print(f"Temps simulé : {r['temps_simule_s']:.1f} s (calcul : {time.perf_counter()-t0:.1f} s)")
    print(f"Erreur d'approche de la pince : {r['erreur_approche_mm']:.2f} mm")
    print(f"Coin soulevé à : {r['hauteur_coin_mm']:.0f} mm au-dessus de la table "
          f"(voisin : {r['hauteur_voisin_mm']:.0f} mm)")
    print(f"Hauteur max du tissu à la fin : {r['hauteur_finale_max_mm']:.1f} mm")
    print(f"Tissu resté sur la table : {r['sur_table']} ; simulation stable : {r['stable']}")
    ok = (r["hauteur_coin_mm"] > 120 and r["sur_table"] and r["stable"]
          and r["erreur_approche_mm"] < 5)
    print("SAISIE :", "RÉUSSIE" if ok else "ÉCHEC")


if __name__ == "__main__":
    main()
