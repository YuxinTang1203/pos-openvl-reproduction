m = robo.move_to([0.20,-0.03,0.90], arm='right', gripper=robo.CLOSE, pitch=0, yaw=78.4, path='auto')
print("rotate away:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.ik_error, m.note, m.steps_used)
robo.show('cam_head')
