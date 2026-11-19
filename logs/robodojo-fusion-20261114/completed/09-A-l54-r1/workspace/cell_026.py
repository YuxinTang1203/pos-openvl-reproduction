import numpy as np
# tube on plate: centre (0.0186,-0.0218), long axis -8.5 deg, length 0.108; cap end at -x side
C=np.array([0.0186,-0.0218]); ang=np.deg2rad(-8.5); u=np.array([np.cos(ang),np.sin(ang)])
cap=C-0.054*u; tip=C+0.054*u
gp=cap+0.02*u   # grasp 2 cm from cap toward tip
print("cap",cap.round(4),"tip",tip.round(4),"grasp",gp.round(4))
m=robo.move_to([gp[0],gp[1],0.90],arm='right',gripper=robo.OPEN,pitch=90,yaw=81.5,path='line')
print("above:",m.reached,m.final_xyz,m.ik_error,m.steps_used)
m2=robo.move_to([gp[0],gp[1],0.845],arm='right',gripper=robo.OPEN,pitch=90,yaw=81.5,path='line')
print("descend:",m2.reached,m2.final_xyz,m2.blocked,m2.ik_error)
g=robo.set_gripper('right',robo.CLOSE)
print("close:",g.holding,g.width_m,g.opening)
m3=robo.move_to([gp[0],gp[1],0.93],arm='right',gripper=robo.CLOSE,pitch=90,yaw=81.5,path='line')
print("lift:",m3.reached,m3.final_xyz,m3.blocked)
