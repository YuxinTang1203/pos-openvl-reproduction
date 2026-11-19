# Make the tube vertical: level gripper pointing away, roll 0 (fingers close along x, tube axis z).
m = robo.move_to([0.072, -0.06, 0.90], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=0, yaw=0, roll=0, path='line')
print('vertical prep:', m.reached, [round(v,4) for v in m.final_xyz], 'pitch', round(m.pitch,1), 'yaw', round(m.yaw,1), 'roll', round(m.roll,1),
      'err', round(m.orientation_error_deg,1), 'ik', m.ik_error, 'blocked', m.blocked, m.steps_used)
robo.show('cam_head')
