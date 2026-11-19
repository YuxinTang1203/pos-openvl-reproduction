m = robo.move_to([0.0742,0.0046,0.895], arm='right', gripper=robo.CLOSE, pitch=0, yaw=76, roll=0, path='line')
print("lower:", m.reached, m.final_xyz, m.blocked, m.ik_error)
im = robo.image('cam_head')
display(im[100:240, 240:420])
