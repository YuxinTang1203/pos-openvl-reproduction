h=robo.home('right')
print("home:",h.reached,h.steps_used)
s=robo.state(); print("right tcp",s.right.tcp_pos,"grip",s.right.gripper)
robo.show('cam_head')
