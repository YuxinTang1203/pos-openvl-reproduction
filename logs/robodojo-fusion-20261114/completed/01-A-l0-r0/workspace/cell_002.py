pts = {}
for name,(r,c) in {"plate":(175,300),"base":(207,320),"table":(230,320),"hole":(183,330),"tube1":(218,211)}.items():
    p = robo.back_project(r,c,'cam_head')
    pts[name]=p
    print(name, p.surface, [round(v,4) for v in p.xyz])
