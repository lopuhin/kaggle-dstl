#!/usr/bin/env python3
import pandas as pd
import shapely.wkt
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon


sample_submission = pd.read_csv('sample_submission.csv')


subm_names = [
    'unet-topscale-4-rot10-channels-12-dice-10-cls-8-9-model-53-eps-1-cls-8',
    'unet-fullaug-square-dice-10-bn-eps-2-cls-8-9',
    'unet-topscale-4-rot25-channels-12-dice-10-cls-8-9-all-eps-1',
]

classes = [9]

buffer_size = 1e-7

min_total_area = 0

print("Loading subms...")

subms = [pd.read_csv('%s.csv.gz' % s) for s in subm_names]

subm = sample_submission.copy()

for image_id in sorted(subm['ImageId'].unique()):
    print("%s..." % image_id)

    subm.loc[subm['ImageId'] == image_id, 'MultipolygonWKT'] = 'MULTIPOLYGON EMPTY'

    for cls in classes:
        polys = [shapely.wkt.loads(s.loc[(s['ImageId'] == image_id) & (s['ClassType'] == cls), 'MultipolygonWKT'].iloc[0]) for s in subms]

        try:
            poly_parts = []

            for i in range(len(polys)):
                for j in range(i+1, len(polys)):
                    poly_parts.append(polys[i].intersection(polys[j]))

            res = unary_union(poly_parts)
        except Exception:
            print("Error, using first poly")
            res = polys[0]

        res = res.buffer(buffer_size, cap_style=3, join_style=3).simplify(1e-6).buffer(0)

        if res.area < min_total_area:
            continue

        if res.type == 'Polygon':
            res = MultiPolygon([res])

        if not res.is_valid:
            raise ValueError("Invalid geometry")

        subm.loc[(subm['ImageId'] == image_id) & (subm['ClassType'] == cls), 'MultipolygonWKT'] = shapely.wkt.dumps(res, rounding_precision=9)

print("Saving...")
subm_name = 'vote-%s-%s' % ('+'.join(map(str, classes)), '+'.join(subm_names))
subm.to_csv('%s.csv.gz' % subm_name, compression='gzip', index=False)

print("Done, %s" % subm_name)
