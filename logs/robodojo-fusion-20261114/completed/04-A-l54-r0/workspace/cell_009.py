
m = robo.move_to([0.2197, -0.0834, 0.90], arm='right', gripper=robo.CLOSE, pitch=0, yaw=-40, roll=0, path='line')
print("rotate:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.blocked, m.ik_error, m.note, m.steps_used)
st = robo.state()
print("opening", st.right.gripper)
robo.show('cam_head')
robo.show('cam_right_wrist')
