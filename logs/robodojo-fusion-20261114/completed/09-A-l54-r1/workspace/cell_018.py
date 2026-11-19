# Tube now hangs vertically (cap up) from the level right gripper. Move it above the chosen hole.
m = robo.move_to([0.0719,0.0058,0.90], arm='right', gripper=robo.CLOSE, pitch=0, yaw=76, roll=0, path='auto')
print("above hole:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.ik_error, m.note, m.blocked, m.steps_used)
robo.show('cam_head')
