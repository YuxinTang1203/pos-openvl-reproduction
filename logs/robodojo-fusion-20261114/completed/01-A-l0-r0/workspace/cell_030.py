t1 = [-0.0942, 0.1356]
m = robo.move_to([t1[0], t1[1], 0.80], arm='left', gripper=robo.OPEN, pitch=60, yaw=-36.1, path='line')
print("approach:", m.reached, m.final_xyz, m.ik_error)
g = robo.set_gripper('left', robo.CLOSE)
print("close:", g.holding, g.width_m)
m2 = robo.move_to([t1[0], t1[1], 0.86], arm='left', gripper=robo.CLOSE, pitch=60, yaw=-36.1, path='line')
print("lift:", m2.reached, m2.final_xyz)
m3 = robo.move_to([t1[0], t1[1], 0.87], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-36.1, path='line')
print("rotate:", m3.reached, m3.final_xyz, m3.pitch)
