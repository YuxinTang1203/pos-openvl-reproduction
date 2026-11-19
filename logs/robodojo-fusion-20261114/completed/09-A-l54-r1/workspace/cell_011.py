# Right arm picks up tube-1 where the left arm placed it
m = robo.move_to([0.12,-0.22,0.87], arm='right', gripper=robo.OPEN, pitch=90, yaw=7.5, path='line')
print("above:", m.reached, m.final_xyz, m.ik_error)
o = robo.locate("white tube with orange cap", camera='cam_right_wrist')
print("wrist:", o.found, o.center_xyz, o.top_z, o.length_m, o.width_m, o.grasp_yaw, o.views, o.score, o.note)
