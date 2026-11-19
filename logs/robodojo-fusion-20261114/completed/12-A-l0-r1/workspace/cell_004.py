import numpy as np

TUBE3 = dict(xyz=(0.340, -0.0517, 0.7721), gyaw=34.8)

# 1) go above tube 3 with right arm, pointing down
m = robo.move_to([0.340, -0.052, 0.87], arm=robo.RIGHT, gripper=robo.OPEN, pitch=90, yaw=34.8, path='line')
print('above:', m.reached, m.final_xyz, m.dist_to_target_m, m.ik_error, m.note, m.steps_used)

# 2) fresh grounding from the right wrist camera (looks down the tube axis)
from qwen_tools import Visual_Grounding_Tool if False else None
