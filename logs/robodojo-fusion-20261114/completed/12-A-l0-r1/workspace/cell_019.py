# Move the right arm clear, then look at the whole scene to find where the released tube went.
m = robo.move_to([0.32, -0.20, 0.93], arm=robo.RIGHT, gripper=robo.OPEN, pitch=90, yaw=0, path='line')
print('clear:', m.reached, [round(v,4) for v in m.final_xyz], m.ik_error, m.steps_used)
robo.show('cam_head')
