
m = robo.move_to([0.12, -0.12, 0.90], arm='right', gripper=robo.CLOSE, pitch=60, yaw=-30, path='auto')
print("recover:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.ik_error, m.note, m.steps_used)
st = robo.state()
print("opening", st.right.gripper)
robo.show('cam_head')
