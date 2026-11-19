seg = robo.segment("white tube with orange cap", camera='cam_head')
print("seg", seg.found, seg.centroid, seg.area_px, seg.score, seg.box)
if seg.found and seg.box is not None:
    b = np.array(seg.box) if not isinstance(seg.box, np.ndarray) else seg.box
    print("box", b)
    # box format [x1,y1,x2,y2] pixels
    x1,y1,x2,y2 = [int(v) for v in list(b)[:4]]
    row = y2-2; col = (x1+x2)//2
    p = robo.back_project(row, col, 'cam_head')
    print("tip pixel", row, col, "surface", p.surface, "xyz", tuple(round(v,4) for v in p.xyz))
    # also top of tube (cap)
    p2 = robo.back_project(y1+2, col, 'cam_head')
    print("cap pixel", y1+2, col, p2.surface, tuple(round(v,4) for v in p2.xyz))
