
gx, gy = 0.0773, -0.0663
m = robo.move_to([gx, gy, 0.788], arm='right', gripper=robo.OPEN, pitch=90, yaw=-58.6, path='line')
print("descend:", m.reached, m.final_xyz, m.blocked, m.steps_used)
g = robo.set_gripper('right', robo.CLOSE)
print("close:", g.holding, g.opening, g.width_m)
if not g.holding:
    # nudge up and try again slightly deeper/other point
    m2 = robo.move_to([0.060, -0.075, 0.788], arm='right', gripper=robo.OPEN, pitch=90, yaw=-58.6, path='line')
    print("nudge:", m2.reached, m2.final_xyz, m2.blocked)
    g = robo.set_gripper('right', robo.CLOSE)
    print("close2:", g.holding, g.opening, g.width_m)
m3 = robo.move_to([gx, gy, 0.87], arm='right', gripper=robo.CLOSE, pitch=90, yaw=-58.6, path='auto')
print("lift:", m3.reached, m3.final_xyz, m3.ik_error)
m4 = robo.move_to([0.0027, 0.0153, 0.845], arm='right', gripper=robo.CLOSE, pitch=60, yaw=-31.1, path='line')
print("to hole:", m4.reached, m4.final_xyz, m4.blocked, m4.ik_error, m4.note)
g2 = robo.set_gripper('right', robo.OPEN)
print("released:", g2.opening)
m5 = robo.move_to([0.0027, 0.0153, 0.93], arm='right', gripper=robo.OPEN, pitch=30, yaw=-31.1, path='line')
print("retreat:", m5.reached, m5.final_xyz, m5.blocked, m5.ik_error, m5.note)
robo.show('cam_head')
