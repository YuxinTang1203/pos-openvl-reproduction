
m = robo.move_to([0.16, -0.10, 0.98], arm='right', gripper=robo.OPEN, pitch=45, yaw=-40, path='auto')
print("away:", m.reached, m.final_xyz, m.steps_used)
robo.show('cam_head')
