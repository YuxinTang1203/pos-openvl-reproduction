# First bring the tube vertical well in front of the rack (clear of it), then probe descent heights.
m = robo.move_to([0.10, -0.10, 0.90], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=90, yaw=0, path='line')
print('prep:', m.reached, [round(v,4) for v in m.final_xyz], m.ik_error, m.steps_used)

for z in (0.86, 0.85, 0.84):
    m = robo.move_to([0.072, 0.045, z], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=90, yaw=0, path='line')
    print('probe', z, 'reached', m.reached, 'final', [round(v,4) for v in m.final_xyz],
          'd', round(m.dist_to_target_m,4), 'ik', m.ik_error, 'blocked', m.blocked, 'steps', m.steps_used)
    if m.reached or m.blocked:
        break
