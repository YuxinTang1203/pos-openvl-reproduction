
# Tube A: center (0.1467,-0.1003) axis 58.9, cap end at (0.176,-0.051). Grasp 1.5cm in from cap end.
gx, gy = 0.1467+0.0427*np.cos(np.radians(58.9)), -0.1003+0.0427*np.sin(np.radians(58.9))
print("grasp", round(gx,4), round(gy,4))
m = robo.move_to([gx, gy, 0.87], arm='right', gripper=robo.OPEN, pitch=90, yaw=-31.1, path='auto')
print("above:", m.reached, m.final_xyz, m.steps_used)
m2 = robo.move_to([gx, gy, 0.787], arm='right', gripper=robo.OPEN, pitch=90, yaw=-31.1, path='line')
print("descend:", m2.reached, m2.final_xyz, m2.blocked, m2.ik_error, m2.steps_used)
g = robo.set_gripper('right', robo.CLOSE)
print("close:", g.holding, g.opening, g.width_m)
m3 = robo.move_to([gx, gy, 0.95], arm='right', gripper=robo.CLOSE, pitch=90, yaw=-31.1, path='line')
print("lift:", m3.reached, m3.final_xyz, m3.blocked, m3.steps_used)
robo.show('cam_head')
