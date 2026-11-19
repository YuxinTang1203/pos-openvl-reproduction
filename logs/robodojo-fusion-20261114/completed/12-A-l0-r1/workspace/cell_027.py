tx, ty, gy = -0.2396, -0.0483, -17.4
m1 = robo.move_to([tx, ty, 0.86], arm=robo.LEFT, gripper=robo.OPEN, pitch=90, yaw=gy, path='line')
print('above:', m1.reached, [round(v,4) for v in m1.final_xyz], m1.ik_error, m1.steps_used)
m2 = robo.move_to([tx, ty, 0.782], arm=robo.LEFT, gripper=robo.OPEN, pitch=90, yaw=gy, path='line')
print('down :', m2.reached, [round(v,4) for v in m2.final_xyz], m2.ik_error, m2.steps_used)
g = robo.set_gripper(robo.LEFT, robo.CLOSE)
print('close:', g.holding, round(g.opening,3), g.width_m)
m3 = robo.move_to([tx, ty, 0.90], arm=robo.LEFT, gripper=robo.CLOSE, pitch=90, yaw=gy, path='line')
print('lift :', m3.reached, [round(v,4) for v in m3.final_xyz], m3.blocked, m3.steps_used)
