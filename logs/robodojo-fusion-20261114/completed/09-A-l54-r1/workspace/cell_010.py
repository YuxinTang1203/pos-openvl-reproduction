# The left arm cannot reach any rack hole (lateral limit), but the right arm can reach the
# near row. So: left arm places tube-1 down in the right arm's working zone, then homes.
m = robo.move_to([0.12,-0.22,0.88], arm='left', gripper=robo.CLOSE, pitch=90, yaw=7.5, path='line')
print("carry:", m.reached, m.final_xyz, m.steps_used, m.blocked, m.ik_error)
m2 = robo.move_to([0.12,-0.22,0.80], arm='left', gripper=robo.CLOSE, pitch=90, yaw=7.5, path='line')
print("lower:", m2.reached, m2.final_xyz, m2.blocked)
g = robo.set_gripper('left', robo.OPEN)
print("open:", g.opening, g.holding)
m3 = robo.move_to([0.12,-0.22,0.95], arm='left', gripper=robo.OPEN, pitch=90, yaw=7.5, path='line')
print("retreat:", m3.reached, m3.final_xyz)
h = robo.home('left')
print("home left:", h.reached, h.max_joint_error_rad, h.steps_used)
robo.show('cam_head')
