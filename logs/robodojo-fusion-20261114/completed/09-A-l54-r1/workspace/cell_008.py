gp = [-0.3025, -0.0938]
m2 = robo.move_to([gp[0], gp[1], 0.782], arm='left', gripper=robo.OPEN, pitch=90, yaw=7.5, path='line')
print("descend:", m2.reached, m2.final_xyz, m2.dist_to_target_m, m2.ik_error, m2.note, m2.steps_used, m2.blocked)
g = robo.set_gripper('left', robo.CLOSE)
print("close:", g.holding, g.width_m, g.opening, g.steps_used)
m3 = robo.move_to([gp[0], gp[1], 0.90], arm='left', gripper=robo.CLOSE, pitch=90, yaw=7.5, path='line')
print("lift:", m3.reached, m3.final_xyz, m3.steps_used, m3.blocked)
s = robo.state()
print("left grip now", s.left.gripper, "tcp", s.left.tcp_pos)
