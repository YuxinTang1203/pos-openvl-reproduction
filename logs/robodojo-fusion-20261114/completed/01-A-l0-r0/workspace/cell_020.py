# Withdraw the left arm clear of the rack, then look again.
w = robo.move_to([-0.30, -0.15, 0.95], arm='left', gripper=robo.OPEN, pitch=0, yaw=0, path='line')
print("withdraw:", w.reached, w.final_xyz, w.steps_used)
robo.show('cam_head')
