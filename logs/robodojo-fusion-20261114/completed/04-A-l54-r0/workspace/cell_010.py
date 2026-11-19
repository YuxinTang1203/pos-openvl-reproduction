
m = robo.move_to([0.2197, -0.0834, 0.93], arm='right', gripper=robo.CLOSE, path='line')
print("lift2:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.steps_used)
st = robo.state()
print("pose", st.right.pitch, st.right.yaw, st.right.roll, "opening", st.right.gripper)
robo.show('cam_head')
robo.show('cam_right_wrist')
