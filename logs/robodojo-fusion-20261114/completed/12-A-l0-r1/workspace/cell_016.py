# Tube is vertical (cap up), gripper level. Move it over the rightmost hole, staying high.
m = robo.move_to([0.072, 0.046, 0.90], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=0, yaw=0, roll=0, path='line')
print('over hole:', m.reached, [round(v,4) for v in m.final_xyz], 'err', round(m.orientation_error_deg,1),
      'ik', m.ik_error, 'blocked', m.blocked, m.steps_used)
robo.show('cam_head')
