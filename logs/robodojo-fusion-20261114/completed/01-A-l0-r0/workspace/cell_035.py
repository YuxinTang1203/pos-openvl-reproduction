# Grasp tube 1 near its cap end so the tube hangs ~9 cm below the grasp point.
gp = [-0.0833, 0.1606]
m = robo.move_to([gp[0], gp[1], 0.80], arm='left', gripper=robo.OPEN, pitch=60, yaw=-33, path='line')
print("approach:", m.reached, m.final_xyz, m.steps_used)
g = robo.set_gripper('left', robo.CLOSE)
print("close:", g.holding, g.width_m)
m2 = robo.move_to([gp[0], gp[1], 0.86], arm='left', gripper=robo.CLOSE, pitch=60, yaw=-33, path='line')
print("lift:", m2.reached, m2.final_xyz)
m3 = robo.move_to([gp[0], gp[1], 0.87], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-33, path='line')
print("rotate:", m3.reached, m3.final_xyz, m3.pitch)
robo.show('cam_head')
