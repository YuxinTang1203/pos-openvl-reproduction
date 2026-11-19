hole=(0.0731,0.0128)
m=robo.move_to([hole[0],hole[1],0.90],arm='right',gripper=robo.CLOSE,pitch=0,yaw=-38.4,roll=0,path='auto')
print("over hole:",m.reached,m.final_xyz,m.ik_error,m.blocked,m.steps_used)
x,y=m.final_xyz[0],m.final_xyz[1]
m2=robo.move_to([x,y,0.893],arm='right',gripper=robo.CLOSE,pitch=0,yaw=-38.4,roll=0,path='line')
print("lower:",m2.reached,m2.final_xyz,m2.blocked,m2.ik_error)
robo.show('cam_head')
