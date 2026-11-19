
import numpy as np
# Tube-2 lying at (0.0534,-0.0809) axis 31.4deg; cap end toward +u
ax = np.radians(31.4); u = np.array([np.cos(ax), np.sin(ax)])
c = np.array([0.0534, -0.0809])
gp = c + 0.028*u   # grasp 1.5cm in from cap end
print("grasp", gp)
m = robo.move_to([float(gp[0]), float(gp[1]), 0.87], arm='right', gripper=robo.OPEN, pitch=90, yaw=-58.6, path='auto')
print("above:", m.reached, m.final_xyz, m.ik_error)
m2 = robo.move_to([float(gp[0]), float(gp[1]), 0.788], arm='right', gripper=robo.OPEN, pitch=90, yaw=-58.6, path='line')
print("desc:", m2.reached, m2.final_xyz, m2.blocked, m2.ik_error)
g = robo.set_gripper('right', robo.CLOSE)
print("close:", g.holding, g.width_m)
m3 = robo.move_to([float(gp[0]), float(gp[1]), 0.87], arm='right', gripper=robo.CLOSE, pitch=90, yaw=-58.6, path='auto')
print("lift:", m3.reached, m3.final_xyz, m3.ik_error, m3.note)
# carry over big hole, tilt to pitch 60 (tube hangs vertical from fingers)
hx, hy = 0.0027, 0.0153
m4 = robo.move_to([hx, hy, 0.87], arm='right', gripper=robo.CLOSE, pitch=60, yaw=-58.6, path='auto')
print("carry:", m4.reached, m4.final_xyz, m4.ik_error, m4.note)
m5 = robo.move_to([hx, hy, 0.845], arm='right', gripper=robo.CLOSE, pitch=60, yaw=-58.6, path='line')
print("descend2:", m5.reached, m5.final_xyz, m5.blocked, m5.ik_error)
g2 = robo.set_gripper('right', robo.OPEN)
print("open:", g2.opening, g2.holding)
m6 = robo.move_to([hx, hy, 0.885], arm='right', gripper=robo.OPEN, pitch=60, yaw=-58.6, path='line')
print("retreat-up:", m6.reached, m6.final_xyz, m6.blocked, m6.ik_error)
robo.show('cam_head')
