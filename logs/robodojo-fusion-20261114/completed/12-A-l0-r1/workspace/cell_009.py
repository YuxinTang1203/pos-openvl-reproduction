# Rack (2-view): plate top z=0.823, holes row at y~0.03-0.05, x from -0.08..0.07
# Right arm holds tube 1 (cap up, body down). Try to reach the right-most hole.
tgt = [0.0722, 0.0496, 0.885]
m = robo.move_to(tgt, arm=robo.RIGHT, gripper=robo.CLOSE, pitch=90, yaw=0, path='line')
print('probe above hole:', m.reached, m.final_xyz, round(m.dist_to_target_m,4), m.ik_error, m.note, m.blocked, m.steps_used)
print('state', robo.state().right.tcp_pos, robo.state().right.pitch, robo.state().right.yaw, robo.state().right.gripper)
