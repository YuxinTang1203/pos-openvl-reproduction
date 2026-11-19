
m = robo.move_to([0.2197, -0.0834, 0.90], arm='right', gripper=robo.CLOSE, pitch=90, yaw=-40, path='line')
print("lift:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.blocked, m.ik_error, m.steps_used)
st = robo.state()
print("gripper opening now:", st.right.gripper)
robo.show('cam_right_wrist')
