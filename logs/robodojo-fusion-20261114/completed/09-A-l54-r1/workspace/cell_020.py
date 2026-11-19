import numpy as np
im = robo.image('cam_head')
crop = im[120:230, 260:420]
display(crop)
