gx, gy = 0.1616, -0.2302
m = robo.move_to([gx,gy,0.782], arm='right', gripper=robo.OPEN, pitch=90, yaw=78.4, path='line')
print("descend:", m.reached, m.final_xyz, m.blocked, m.ik_error)
g = robo.set_gripper('right', robo.CLOSE)
print("close:", g.holding, g.width_m, g.opening)
m2 = robo.move_to([gx,gy,0.95], arm='right', gripper=robo.CLOSE, pitch=90, yaw=78.4, path='line')
print("lift:", m2.reached, m2.final_xyz, m2.blocked)
robo.show('cam_right_wrist')
