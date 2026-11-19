
g = robo.set_gripper('right', robo.OPEN)
print("open:", g.opening, g.holding, g.steps_used)
m = robo.move_to([0.0149, 0.0058, 0.95], arm='right', gripper=robo.OPEN, path='line')
print("retreat:", m.reached, m.final_xyz, m.blocked, m.steps_used)
robo.show('cam_head')
