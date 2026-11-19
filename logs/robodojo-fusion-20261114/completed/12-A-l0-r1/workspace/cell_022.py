# Reorient the held tube vertical away from the rack, then insert into the next hole.
m = robo.move_to([0.10, -0.08, 0.90], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=0, yaw=0, roll=0, path='line')
print('vertical:', m.reached, [round(v,4) for v in m.final_xyz], 'pitch', round(m.pitch,1), 'roll', round(m.roll,1), m.ik_error, m.steps_used)

m = robo.move_to([0.032, 0.047, 0.90], arm=robo.RIGHT, gripper=robo.CLOSE, pitch=0, yaw=0, roll=0, path='line')
print('over hole2:', m.reached, [round(v,4) for v in m.final_xyz], m.ik_error, m.blocked, m.steps_used)
