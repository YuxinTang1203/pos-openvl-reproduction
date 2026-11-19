g=robo.set_gripper('left', robo.OPEN); print("open:",g.opening, g.steps_used)
h=robo.home('both'); print("home:", h.reached, h.max_joint_error_rad, h.steps_used)
s=robo.state(); print("L",s.left.tcp_pos, s.left.gripper, "R", s.right.tcp_pos, s.right.gripper, "trunc", s.truncated)
