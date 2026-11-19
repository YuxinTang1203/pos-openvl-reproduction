robo.show('cam_left_wrist')
import numpy as np
s=robo.state()
print("L tcp", s.left.tcp_pos, s.left.pitch, s.left.yaw, "grip", s.left.gripper)
