cands = [[0.12,0.02,0.87],[0.08,0.02,0.87],[0.08,0.00,0.87],[0.05,0.02,0.87],[0.05,0.00,0.87],[0.02,0.00,0.87]]
for c in cands:
    m = robo.move_to(c, arm=robo.RIGHT, gripper=robo.CLOSE, pitch=90, yaw=0, path='line')
    print(c, 'reached', m.reached, 'final', [round(v,4) for v in m.final_xyz], 'ik', m.ik_error, 'steps', m.steps_used)
    if m.reached:
        break
