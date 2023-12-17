cross_ABCD_kwargs = {'bottom':0.005,'slope':0.3,'ax':0.625, 'ay':0.625, 'bx':0.875, 'by':0.875, 'cx':0.125, "cy":0.375, 'dx':0.375, 'dy':0.125}
width_0 = lambda mid_pt: cross_ABCD(mid_pt, **cross_ABCD_kwargs)
width_1 = width_0
height_0 = lambda mid_pt: cross_ABCD(mid_pt, **cross_ABCD_kwargs)
height_1 = height_0