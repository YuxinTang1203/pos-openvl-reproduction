
# Move held tube over big-hole target (0.0027, 0.0153), keep pitch 45, then descend stepwise.
m = robo.move_to([0.0027, 0.0153, 0.95], arm='right', gripper=robo.CLOSE, pitch=45, yaw=-40, path='line')
print("over hole:", m.reached, m.final_xyz, m.pitch, m.yaw, m.blocked, m.ik_error, m.steps_used)
m2 = robo.move_to([0.0027, 0.0153, 0.905], arm='right', gripper=robo.CLOSE, pitch=45, yaw=-40, path='line')
print("descend1:", m2.reached, m2.final_xyz, m2.blocked, m2.ik_error, m2.steps_used)
robo.show('cam_head')
