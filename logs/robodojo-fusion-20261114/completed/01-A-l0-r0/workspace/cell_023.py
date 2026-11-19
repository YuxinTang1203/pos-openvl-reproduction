t1 = [-0.090, 0.139]
m = robo.move_to([t1[0], t1[1], 0.86], arm='left', gripper=robo.OPEN, pitch=90, yaw=-30, path='line')
print("above:", m.reached, m.final_xyz, m.ik_error, m.steps_used)
m2 = robo.move_to([t1[0], t1[1], 0.795], arm='left', gripper=robo.OPEN, pitch=90, yaw=-30, path='line')
print("descend:", m2.reached, m2.final_xyz, m2.ik_error, m2.blocked, m2.steps_used)
g = robo.set_gripper('left', robo.CLOSE)
print("close:", g.holding, g.width_m)
m3 = robo.move_to([t1[0], t1[1], 0.90], arm='left', gripper=robo.CLOSE, pitch=90, yaw=-30, path='line')
print("lift:", m3.reached, m3.final_xyz, m3.ik_error, m3.steps_used)
