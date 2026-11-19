m = robo.move_to([0.0742,0.0051,0.865], arm='right', gripper=robo.CLOSE, pitch=0, yaw=76, roll=0, path='line')
print("descend1:", m.reached, m.final_xyz, m.blocked, m.ik_error, m.steps_used)
robo.show('cam_head')
