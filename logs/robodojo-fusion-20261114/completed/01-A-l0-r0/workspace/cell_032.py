s = robo.state()
print("step", s.step, "trunc", s.truncated)
g = robo.set_gripper('left', robo.OPEN)  # spends ~2 actions, gives step/limit
print(g.step, g.step_limit)
import numpy as np
from collections import deque
img = robo.image('cam_head').astype(int)
B=img[:,:,2]; Rr=img[:,:,0]
blue = (B > Rr+25) & (B > 110)
ys,xs = np.nonzero(blue)
print("blue rows", ys.min(), ys.max(), "cols", xs.min(), xs.max())
