# Make the held tube vertical: level gripper (pitch 0) with roll 0 -> tube axis vertical
m = robo.move_to([0.20,-0.05,0.92], arm='right', gripper=robo.CLOSE, pitch=0, yaw=78.4, roll=0, path='auto')
print("roll0:", m.reached, m.final_xyz, m.pitch, m.yaw, m.roll, m.ik_error, m.note, m.steps_used)
robo.show('cam_head')
