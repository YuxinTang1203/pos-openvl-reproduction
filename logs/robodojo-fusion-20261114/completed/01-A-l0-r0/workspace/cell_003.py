import numpy as np
# Grasp tube 1 with the left arm: position above, then descend along the tube's narrow side.
t1 = np.array([-0.240, -0.049])
m1 = robo.move_to([t1[0], t1[1], 0.87], arm='left', gripper=robo.KEEP, pitch=90, yaw=-17.2, path='line')
print("above:", m1.reached, m1.final_xyz, m1.ik_error, m1.note, m1.steps_used)
m2 = robo.move_to([t1[0], t1[1], 0.786], arm='left', gripper=robo.KEEP, pitch=90, yaw=-17.2, path='line')
print("descend:", m2.reached, m2.final_xyz, m2.ik_error, m2.note, m2.steps_used)
g = robo.set_gripper('left', robo.CLOSE)
print("close:", g.holding, g.width_m, g.opening, g.steps_used, g.truncated)
