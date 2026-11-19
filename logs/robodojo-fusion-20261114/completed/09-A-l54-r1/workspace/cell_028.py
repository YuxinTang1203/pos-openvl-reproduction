robo.show('cam_right_wrist')
s=robo.state()
print("right tcp", s.right.tcp_pos, s.right.pitch, s.right.yaw, s.right.roll, "grip", s.right.gripper)
