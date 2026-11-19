h = [-0.0422, 0.0599]
m1 = robo.move_to([h[0], h[1], 0.875], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-36.1, path='line')
print("align:", m1.reached, m1.final_xyz)
m2 = robo.move_to([h[0], h[1], 0.845], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-36.1, path='line')
print("insert:", m2.reached, m2.final_xyz, m2.blocked, m2.steps_used)
g = robo.set_gripper('left', robo.OPEN)
m3 = robo.move_to([h[0], h[1], 0.95], arm='left', gripper=robo.OPEN, pitch=0, yaw=-36.1, path='line')
print("retreat:", m3.reached, m3.final_xyz)
robo.show('cam_head')
