# Tube is now vertical in the right gripper (axis along z, tip ~0.059 m below the grasp point).
# Probe the pitch=90 envelope at z=0.89 (tube tip 8 mm above the plate top 0.823) toward the holes.
for tgt in [[0.14,0.06,0.89],[0.10,0.06,0.89],[0.072,0.050,0.89]]:
    m = robo.move_to(tgt, arm=robo.RIGHT, gripper=robo.CLOSE, pitch=90, yaw=0, path='line')
    print(tgt, 'reached', m.reached, 'final', [round(v,4) for v in m.final_xyz],
          'd', round(m.dist_to_target_m,4), 'ik', m.ik_error, 'blocked', m.blocked, 'steps', m.steps_used)
    if not m.reached and m.ik_error:
        break
