im = robo.image('cam_head')
print(im.shape)
# crop the rack region for a closer look (head image, rack around cols 250-390, rows 150-230)
import numpy as np
crop = im[140:240, 240:400]
display(crop)
