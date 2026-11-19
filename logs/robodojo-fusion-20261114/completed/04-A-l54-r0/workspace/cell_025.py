
g = robo.set_gripper('right', robo.OPEN)
print("open:", g.opening, g.holding)
m = robo.move_to([0.006, 0.0123, 0.95], arm='right', gripper=robo.OPEN, pitch=60, yaw=-31.1, path='line')
print("retreat:", m.reached, m.final_xyz, m.blocked, m.steps_used)
m2 = robo.move_to([0.20, -0.12, 0.97], arm='right', gripper=robo.OPEN, pitch=60, yaw=-31.1, path='auto')
print("away:", m2.reached, m2.final_xyz, m2.steps_used)
robo.show('cam_head')
