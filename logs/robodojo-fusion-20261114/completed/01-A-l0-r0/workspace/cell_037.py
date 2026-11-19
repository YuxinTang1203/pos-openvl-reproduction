gp=[-0.0833,0.1606]
mA=robo.move_to([gp[0],gp[1],0.80],arm='left',gripper=robo.OPEN,pitch=60,yaw=-33,path='line')
print("approach:",mA.reached,mA.final_xyz,mA.steps_used)
gA=robo.set_gripper('left',robo.CLOSE); print("close:",gA.holding,gA.width_m)
mB=robo.move_to([gp[0],gp[1],0.87],arm='left',gripper=robo.CLOSE,pitch=0,yaw=-33,path='line')
print("rotate:",mB.reached,mB.final_xyz,mB.steps_used)
