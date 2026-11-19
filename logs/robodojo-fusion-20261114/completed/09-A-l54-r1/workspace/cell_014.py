# Rotate wrist to level so the held tube hangs vertically (cap end up or down TBD)
m = robo.move_to([0.1616,-0.2302,0.95], arm='right', gripper=robo.CLOSE, pitch=0, yaw=78.4, path='line')
print("rotate:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.orientation_error_deg, m.blocked, m.ik_error, m.steps_used)
robo.show('cam_head')
