# Measured now: plate top z~0.815; far-row right hole at (0.0758, 0.0214); near-row right at (0.0721,-0.0422).
# My earlier target (0.0719,0.0058) was BETWEEN the rows - that is why the tube rested on the plate.
m=robo.move_to([0.0758,0.0214,0.895],arm='right',gripper=robo.CLOSE,pitch=0,yaw=76,roll=0,path='line')
print("over far hole:",m.reached,m.final_xyz,m.blocked,m.ik_error)
m2=robo.move_to([0.0758,0.0214,0.882],arm='right',gripper=robo.CLOSE,pitch=0,yaw=76,roll=0,path='line')
print("lower:",m2.reached,m2.final_xyz,m2.blocked,m2.ik_error)
robo.show('cam_head')
