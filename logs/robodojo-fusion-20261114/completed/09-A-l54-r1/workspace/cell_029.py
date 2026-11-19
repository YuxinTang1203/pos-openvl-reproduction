m = robo.move_to([-0.0129,-0.0213,0.97], arm='right', gripper=robo.CLOSE, pitch=0, yaw=76, roll=0, path='auto')
print("vertical hang:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.ik_error, m.note, m.steps_used)
robo.show('cam_head')
