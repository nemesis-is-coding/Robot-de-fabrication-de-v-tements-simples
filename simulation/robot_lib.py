"""Bibliothèque commune : scène MuJoCo (table, tissu, robot à ventouse) et cinématique inverse.

Le robot est un bras à 4 axes (lacet, épaule, coude, poignet) équipé d'une plaque à aspiration.
La saisie du tissu est virtuelle : 36 sommets répartis sous la plaque sont liés à la pince.
"""
import time

import mujoco
import numpy as np

# Géométrie du bras (m)
BASE = (0.0, -0.65)             # position (x, y) de l'axe du robot
Z_EPAULE = 0.42                 # hauteur de l'épaule
L1, L2, LP = 0.45, 0.42, 0.14   # bras, avant-bras, longueur poignet+ventouse
PAS = 0.04                      # maille du tissu
NX, NY = 11, 12                 # sommets du tissu : 0.40 x 0.44 m
Z_TABLE = 0.40
# Sommets saisis par la plaque à aspiration (indice = ix * NY + iy) : grille 6 x 6
# répartie sur toute la pièce, symétrique par rapport à son centre
SOMMETS_VENTOUSE = [ix * NY + iy for ix in (0, 2, 4, 6, 8, 10)
                    for iy in (0, 2, 4, 7, 9, 11)]


def scene_xml(masse=0.1):
    liaisons = "\n".join(
        f'    <connect name="v{k}" body1="pince" body2="tissu_{i}" '
        f'anchor="0 {LP} 0" active="false"/>'
        for k, i in enumerate(SOMMETS_VENTOUSE))
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
    <flexcomp type="grid" count="{NX} {NY} 1" spacing="{PAS} {PAS} {PAS}"
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
              <joint damping="2" armature="0.05" name="roulis" type="hinge" axis="0 1 0" range="-3.0 3.0"/>
              <geom type="capsule" fromto="0 0 0 0 {LP-0.01} 0" size="0.012" rgba="0.3 0.3 0.35 1"/>
              <!-- plaque à aspiration de la taille de la pièce (0.40 x 0.44 m) -->
              <geom type="box" size="0.20 0.004 0.22" pos="0 {LP} 0" rgba="0.15 0.15 0.15 0.35"/>
              <site name="pointe" pos="0 {LP} 0" size="0.004"/>
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
    <position name="a_roulis" joint="roulis" kp="100" kv="10"/>
  </actuator>
  <equality>
{liaisons}
  </equality>
</mujoco>
"""


JOINTS = ["lacet", "epaule", "coude", "poignet", "roulis"]


class IK:
    """Cinématique inverse numérique : position de la pointe, pince verticale,
    et plaque toujours orientée selon les axes du monde (grâce au roulis)."""

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
            axe_x = d.site_xmat[self.site].reshape(3, 3)[:, 0]  # côté de la plaque (x local)
            e_rot = (np.cross(axe, np.array([0.0, 0.0, -1.0]))   # pince vers le bas
                     + np.cross(axe_x, np.array([1.0, 0.0, 0.0])))  # plaque alignée sur x
            if np.linalg.norm(e_pos) < 1e-5 and np.linalg.norm(e_rot) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.vadr], jacr[:, self.vadr]])
            e = np.concatenate([e_pos, e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + 0.5 * dq, self.lo, self.hi)
        erreur = np.linalg.norm(cible - pos)
        return q, erreur


def min_jerk(t):
    """Interpolation douce de 0 à 1."""
    t = np.clip(t, 0, 1)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class Robot:
    """Scène complète + commandes de haut niveau (déplacement, saisie, relâchement)."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(scene_xml())
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.ik = IK(m)
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_" + n)
                    for n in JOINTS]
        self.eq = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"v{k}")
                   for k in range(len(SOMMETS_VENTOUSE))]
        self.body_pince = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pince")
        self.body_sommet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"tissu_{i}")
                            for i in SOMMETS_VENTOUSE]
        self.q = np.array([0.0, 0.6, -1.2, -0.9, 0.0])   # posture de départ
        self.d.qpos[self.ik.qadr] = self.q
        self.commander(self.q)
        self.sync = None                                # fonction appelée pour afficher

    def commander(self, q):
        self.d.ctrl[self.act] = q

    def avancer(self, duree, q_fin=None, temps_reel=False):
        """Avance de `duree` secondes en interpolant vers la posture q_fin."""
        q_deb = self.q.copy()
        pas = self.m.opt.timestep
        n = int(duree / pas)
        for i in range(n):
            if q_fin is not None:
                self.q = q_deb + (q_fin - q_deb) * min_jerk((i + 1) / n)
            self.commander(self.q)
            tic = time.perf_counter()
            mujoco.mj_step(self.m, self.d)
            if self.sync and i % 10 == 0:
                self.sync()
                if temps_reel:
                    reste = 10 * pas - (time.perf_counter() - tic)
                    if reste > 0:
                        time.sleep(reste)

    def vers(self, cible, duree, temps_reel=False):
        q, err = self.ik.resoudre(np.array(cible), self.q)
        if err > 2e-3:
            raise RuntimeError(f"cible hors de portée ({err*1000:.0f} mm) : {np.round(cible, 3)}")
        self.avancer(duree, q, temps_reel)

    def saisir(self):
        """Lie les sommets du tissu sous la ventouse à la pince."""
        d, m = self.d, self.m
        mujoco.mj_forward(m, d)
        pince = d.xpos[self.body_pince]
        R = d.xmat[self.body_pince].reshape(3, 3)
        for eq, b in zip(self.eq, self.body_sommet):
            m.eq_data[eq, 0:3] = R.T @ (d.xpos[b] - pince)   # ancre dans le repère de la pince
            m.eq_data[eq, 3:6] = 0.0
            d.eq_active[eq] = 1

    def relacher(self):
        for eq in self.eq:
            self.d.eq_active[eq] = 0

    def pointe(self):
        return self.d.site_xpos[self.ik.site].copy()

    def sommets(self):
        """Positions (NX*NY, 3) des sommets du tissu."""
        return self.d.flexvert_xpos.copy()
