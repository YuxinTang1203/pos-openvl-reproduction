# Tube hangs vertically right above the rightmost hole. Lower it into the hole in small line steps.
for z in (0.890, 0.882, 0.876):
    m = robo.move_to([0.072, 0.044, z], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=0, yaw=0, roll=0, path='line')
    print('descend', z, 'reached', m.reached, 'final', [round(v,4) for v in m.final_xyz],
          'ik', m.ik_error, 'blocked', m.blocked, 'steps', m.steps_used)
    if m.blocked: break

g = robo.set_gripper(robo.RIGHT, robo.OPEN)
print('open:', g.commanded, 'opening', round(g.opening,3), 'holding', g.holding, g.steps_used)

m = robo.move_to([0.072, 0.044, 0.95], arm=robo.RIGHT, gripper=robo.OPEN, pitch=0, yaw=0, roll=0, path='line')
print('retract:', m.reached, [round(v,4) for v in m.final_xyz], m.ik_error, m.steps_used)
robo.show('cam_head')
