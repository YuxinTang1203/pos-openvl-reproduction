print(robo.task)
st = robo.state()
print("truncated", st.truncated)
print("left tcp", st.left.tcp_pos, st.left.pitch, st.left.yaw, st.left.roll, st.left.gripper)
print("right tcp", st.right.tcp_pos, st.right.pitch, st.right.yaw, st.right.roll, st.right.gripper)

rack = robo.locate("blue test tube rack")
print("rack", rack)
