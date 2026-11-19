# Transport tube 1 over a far-row (large) hole of the rack, keeping the tube vertical (pitch=0).
tgt = [-0.05, 0.071, 0.88]
m = robo.move_to(tgt, arm='left', gripper=robo.CLOSE, pitch=0, yaw=-17.2, path='line')
print("transport:", m.reached, m.final_xyz, m.steps_used, m.blocked, m.ik_error)
robo.show('cam_head')
