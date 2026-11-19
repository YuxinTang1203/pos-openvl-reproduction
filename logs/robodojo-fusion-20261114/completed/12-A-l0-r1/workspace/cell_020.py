img = robo.image('cam_head')
crop = img[120:200, 300:420]
display(crop)
# also measure the standing tube with locate (candidate-based to avoid ambiguity)
try:
    o = robo.locate("test tube standing upright in the rack")
    print('stand:', o.center_xyz, 'top', o.top_z, 'h', o.height_m, 'views', o.views, 'score', o.score, 'note', o.note)
except Exception as e:
    print('loc exc', str(e)[:200])
