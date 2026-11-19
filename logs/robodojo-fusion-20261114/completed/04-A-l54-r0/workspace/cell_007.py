
import numpy as np
# refined tube-2: center (0.1948,-0.1146,0.7771), axis 51.5 deg, length 0.1088 -> grasp 4cm toward cap end
gx, gy = 0.1948+0.04*np.cos(np.radians(51.5)), -0.1146+0.04*np.sin(np.radians(51.5))
print("grasp xy", round(gx,4), round(gy,4))

m = robo.move_to([gx, gy, 0.786], arm='right', gripper=robo.OPEN, pitch=90, yaw=-40, path='line')
print("descend:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.ik_error, m.note, m.blocked, m.steps_used)

g = robo.set_gripper('right', robo.CLOSE)
print("close:", g.holding, g.opening, g.width_m, g.steps_used, g.truncated)
