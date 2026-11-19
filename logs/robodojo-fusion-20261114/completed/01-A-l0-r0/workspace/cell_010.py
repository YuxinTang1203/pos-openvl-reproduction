# Align above far-row hole at (-0.0431, 0.0677), then descend to insert tube 1.
h = (-0.0431, 0.0677)
m1 = robo.move_to([h[0], h[1], 0.88], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-17.2, path='line')
print("align:", m1.reached, m1.final_xyz, m1.steps_used)
m2 = robo.move_to([h[0], h[1], 0.838], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-17.2, path='line')
print("descend:", m2.reached, m2.final_xyz, m2.steps_used, m2.blocked, m2.ik_error)
robo.show('cam_head')
