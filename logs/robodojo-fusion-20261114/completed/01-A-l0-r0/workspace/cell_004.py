# Lift tube 1 straight up, then rotate gripper to horizontal (pitch=0) so the tube hangs vertically.
m = robo.move_to([-0.240, -0.049, 0.95], arm='left', gripper=robo.CLOSE, pitch=90, yaw=-17.2, path='line')
print("lift:", m.reached, m.final_xyz, m.steps_used, m.blocked)
m2 = robo.move_to([-0.240, -0.049, 0.95], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-17.2, path='line')
print("rotate:", m2.reached, m2.final_xyz, m2.pitch, m2.yaw, m2.steps_used, m2.note, m2.ik_error)
robo.show('cam_left_wrist')
