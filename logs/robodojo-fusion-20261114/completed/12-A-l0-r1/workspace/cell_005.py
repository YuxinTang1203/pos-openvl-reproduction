tx, ty = 0.340, -0.052
gy = 34.8

m1 = robo.move_to([tx, ty, 0.86], arm=robo.RIGHT, gripper=robo.OPEN, pitch=90, yaw=gy, path='line')
print('above:', m1.reached, m1.final_xyz, round(m1.dist_to_target_m,4), m1.ik_error, m1.note, m1.steps_used)

m2 = robo.move_to([tx, ty, 0.782], arm=robo.RIGHT, gripper=robo.OPEN, pitch=90, yaw=gy, path='line')
print('down :', m2.reached, m2.final_xyz, round(m2.dist_to_target_m,4), m2.ik_error, m2.note, m2.steps_used)

g = robo.set_gripper(robo.RIGHT, robo.CLOSE)
print('close:', g.commanded, 'opening', round(g.opening,3), 'holding', g.holding, 'width', g.width_m, g.steps_used)

m3 = robo.move_to([tx, ty, 0.90], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=90, yaw=gy, path='line')
print('lift :', m3.reached, m3.final_xyz, round(m3.dist_to_target_m,4), m3.blocked, m3.steps_used)
robo.show('cam_head')
