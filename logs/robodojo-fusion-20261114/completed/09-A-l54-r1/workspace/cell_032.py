# Tube is held near its cap (wrist view: cap above fingers) -> tip hangs ~9-10 cm below grasp point.
# 1) rotate to vertical hang, 2) move over the near-row right hole, 3) lower so the tip enters the hole.
m=robo.move_to([-0.0129,-0.0213,0.97],arm='right',gripper=robo.CLOSE,pitch=0,yaw=76,roll=0,path='auto')
print("rotate:",m.reached,m.final_xyz,m.pitch,m.yaw,m.roll,m.ik_error,m.steps_used)
m2=robo.move_to([0.0719,0.0058,0.97],arm='right',gripper=robo.CLOSE,pitch=0,yaw=76,roll=0,path='auto')
print("over hole:",m2.reached,m2.final_xyz,m2.dist_to_target_m,m2.blocked,m2.ik_error,m2.steps_used)
m3=robo.move_to([m2.final_xyz[0],m2.final_xyz[1],0.905],arm='right',gripper=robo.CLOSE,pitch=0,yaw=76,roll=0,path='line')
print("lower:",m3.reached,m3.final_xyz,m3.blocked,m3.ik_error)
robo.show('cam_head')
