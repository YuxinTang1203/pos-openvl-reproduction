# Tube tip rests on solid plate 2.8 cm +y of hole3 (-0.0422,0.0599). Lift, shift so tip is over hole, descend.
up = robo.move_to([-0.0433, 0.0700, 0.885], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-17.2, path='line')
print("up:", up.reached, up.final_xyz, up.steps_used)
mv = robo.move_to([-0.0489, 0.0422, 0.885], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-17.2, path='line')
print("shift:", mv.reached, mv.final_xyz, mv.steps_used)
dn = robo.move_to([-0.0489, 0.0422, 0.852], arm='left', gripper=robo.CLOSE, pitch=0, yaw=-17.2, path='line')
print("descend:", dn.reached, dn.final_xyz, dn.blocked, dn.steps_used)
robo.show('cam_left_wrist')
