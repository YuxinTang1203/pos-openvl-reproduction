# Move above the nearest rack hole with the vertically hanging tube; try pitch 90 lower z first via auto swing
m = robo.move_to([0.0719,0.0058,0.86], arm='right', gripper=robo.CLOSE, pitch=90, yaw=78.4, path='auto')
print("A:", m.reached, m.final_xyz, m.pitch, m.yaw, m.dist_to_target_m, m.ik_error, m.note, m.blocked, m.steps_used)
