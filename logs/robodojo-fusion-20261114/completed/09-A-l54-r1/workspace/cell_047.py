import numpy as np
C=np.array([0.1945,-0.1135]); ax=np.deg2rad(51.6); u=np.array([np.cos(ax),np.sin(ax)])
cap=C+0.058*u; gp=cap-0.02*u
print("cap",cap.round(4),"grasp",gp.round(4))
m=robo.move_to([gp[0],gp[1],0.87],arm='right',gripper=robo.OPEN,pitch=90,yaw=-38.4,path='line')
print("above:",m.reached,m.final_xyz,m.ik_error)
m2=robo.move_to([gp[0],gp[1],0.784],arm='right',gripper=robo.OPEN,pitch=90,yaw=-38.4,path='line')
print("descend:",m2.reached,m2.final_xyz,m2.blocked,m2.ik_error)
g=robo.set_gripper('right',robo.CLOSE)
print("close:",g.holding,g.width_m,g.opening)
if g.holding:
    m3=robo.move_to([gp[0],gp[1],0.95],arm='right',gripper=robo.CLOSE,pitch=90,yaw=-38.4,path='line')
    print("lift:",m3.reached,m3.final_xyz,m3.blocked)
    m4=robo.move_to([0.20,-0.10,0.95],arm='right',gripper=robo.CLOSE,pitch=0,yaw=-38.4,roll=0,path='auto')
    print("vertical:",m4.reached,m4.final_xyz,m4.pitch,m4.roll,m4.ik_error)
robo.show('cam_head')
