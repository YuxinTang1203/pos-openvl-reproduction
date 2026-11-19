
gx, gy = 0.1688, -0.0639
m1 = robo.move_to([gx, gy, 0.92], arm='right', gripper=robo.CLOSE, pitch=90, yaw=-31.1, path='auto')
print("lift:", m1.reached, m1.final_xyz, m1.pitch, m1.steps_used)
m2 = robo.move_to([0.0027, 0.0153, 0.92], arm='right', gripper=robo.CLOSE, pitch=60, yaw=-31.1, path='auto')
print("carry:", m2.reached, m2.final_xyz, m2.pitch, m2.ik_error, m2.blocked, m2.steps_used)
m3 = robo.move_to([0.0027, 0.0153, 0.845], arm='right', gripper=robo.CLOSE, pitch=60, yaw=-31.1, path='line')
print("descend:", m3.reached, m3.final_xyz, m3.blocked, m3.ik_error, m3.note, m3.steps_used)
robo.show('cam_head')
