g0 = robo.set_gripper('left', robo.OPEN)
print("open:", g0.opening)
tgt = [-0.090, 0.139, 0.80]
m = robo.move_to(tgt, arm='left', gripper=robo.OPEN, pitch=60, yaw=-30, path='line')
print("approach:", m.reached, m.final_xyz, m.ik_error, m.steps_used)
