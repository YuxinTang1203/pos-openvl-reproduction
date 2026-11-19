img = robo.image('cam_head')
# crop the rack region in pixel coords: rows 140-230, cols 250-400
crop = img[140:235, 240:410]
display(crop)
print(crop.shape)
