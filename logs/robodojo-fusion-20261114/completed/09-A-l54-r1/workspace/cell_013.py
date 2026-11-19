# Test reach to the rightmost near-row hole while holding the tube vertically
tgt = [0.0719, 0.0058, 0.89]
m = robo.move_to(tgt, arm='right', gripper=robo.CLOSE, pitch=90, yaw=78.4, path='line')
print("try hole:", m.reached, m.final_xyz, m.dist_to_target_m, m.ik_error, m.note, m.steps_used, m.blocked)
