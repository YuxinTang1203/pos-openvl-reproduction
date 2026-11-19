m = robo.move_to([0.0742,0.005,0.93], arm='right', gripper=robo.CLOSE, pitch=0, yaw=76, roll=0, path='line')
print("raise:", m.reached, m.final_xyz, m.blocked)
im = robo.image('cam_head')
display(im[120:230, 250:420])
