
g = robo.set_gripper('right', robo.OPEN)
m = robo.move_to([0.10, -0.05, 0.97], arm='right', gripper=robo.OPEN, pitch=45, yaw=-40, path='auto')
print("back:", m.reached, m.final_xyz)
robo.show('cam_head')
