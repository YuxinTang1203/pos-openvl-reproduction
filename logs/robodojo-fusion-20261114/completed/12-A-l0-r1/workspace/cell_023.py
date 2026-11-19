# Second tube is vertical above hole (0.032,0.047). Lower, release, retract.
for z in (0.890, 0.883, 0.877):
    m = robo.move_to([0.032, 0.047, z], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=0, yaw=0, roll=0, path='line')
    print('desc', z, m.reached, [round(v,4) for v in m.final_xyz], m.blocked, m.steps_used)
g = robo.set_gripper(robo.RIGHT, robo.OPEN)
print('release:', g.holding, round(g.opening,3))
m = robo.move_to([0.032, 0.047, 0.95], arm=robo.RIGHT, gripper=robo.OPEN, pitch=0, yaw=0, roll=0, path='line')
print('retract:', m.reached, [round(v,4) for v in m.final_xyz], m.steps_used)
robo.show('cam_head')
