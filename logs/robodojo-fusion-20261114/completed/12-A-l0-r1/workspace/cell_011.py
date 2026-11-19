st = robo.state()
print('right opening', round(st.right.gripper,3), 'tcp', [round(v,4) for v in st.right.tcp_pos])
print('left  opening', round(st.left.gripper,3), 'tcp', [round(v,4) for v in st.left.tcp_pos])

# Probe the LEFT arm (empty) for a vertical pose over the rack holes.
for tgt in [[-0.040, 0.038, 0.880], [-0.005, 0.041, 0.880], [0.032, 0.047, 0.880]]:
    m = robo.move_to(tgt, arm=robo.LEFT, gripper=robo.OPEN, pitch=90, yaw=0, path='line')
    print('L probe', tgt, 'reached', m.reached, 'final', [round(v,4) for v in m.final_xyz],
          'err', round(m.orientation_error_deg,1), 'ik', m.ik_error, 'steps', m.steps_used)
