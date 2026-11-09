import time
import mujoco

XML = """
<mujoco>
  <option timestep="0.002" solver="CG" tolerance="1e-6"/>
  <size memory="20M"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 .1"/>
    <flexcomp type="grid" count="20 20 1" spacing="0.05 0.05 0.05"
              pos="0 0 0.5" radius="0.002" name="cloth" dim="2" mass="0.2">
      <contact selfcollide="none"/>
      <edge equality="true"/>
    </flexcomp>
  </worldbody>
</mujoco>
"""
open("cloth.xml", "w").write(XML)

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

sim_time = 2.0
steps = int(sim_time / model.opt.timestep)
t0 = time.perf_counter()
for _ in range(steps):
    mujoco.mj_step(model, data)
wall = time.perf_counter() - t0

print(f"Sommets : {model.nflexvert}")
print(f"Temps simulé : {sim_time:.1f} s, temps de calcul : {wall:.2f} s")
print(f"Facteur temps réel : {sim_time / wall:.1f}x")
