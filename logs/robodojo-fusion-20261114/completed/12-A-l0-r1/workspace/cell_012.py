# The tube is held with its axis horizontal along +y (yaw=0 -> fingers close along x).
# Rotating pitch 90 -> 0 about the x-axis should make the tube axis VERTICAL.
m = robo.move_to([0.06, -0.02, 0.93], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=0, yaw=0, path='line')
print('rotate:', m.reached, [round(v,4) for v in m.final_xyz], 'orient_err', round(m.orientation_error_deg,1),
      'pitch', round(m.pitch,1), 'yaw', round(m.yaw,1), 'roll', round(m.roll,1), 'ik', m.ik_error, 'blocked', m.blocked, m.steps_used)
print('gripper opening', round(robo.state().right.gripper,3))
robo.show('cam_head')
