import csv
from itertools import islice
from typing import Dict, Tuple
import sys

import cv2
import numpy as np
import tifffile as tiff
from shapely.geometry import MultiPolygon
import shapely.wkt
import shapely.affinity


csv.field_size_limit(sys.maxsize)


_x_max_y_min = None
_wkt_data = None


def get_x_max_y_min(im_id: str) -> Tuple[float, float]:
    global _x_max_y_min
    if _x_max_y_min is None:
        with open('./grid_sizes.csv') as f:
            _x_max_y_min = {im_id: (float(x), float(y))
                          for im_id, x, y in islice(csv.reader(f), 1, None)}
    return _x_max_y_min[im_id]


def get_wkt_data() -> Dict[str, Dict[int, str]]:
    global _wkt_data
    if _wkt_data is None:
        _wkt_data = {}
        with open('./train_wkt_v4.csv') as f:
            for im_id, poly_type, poly in islice(csv.reader(f), 1, None):
                _wkt_data.setdefault(im_id, {})[int(poly_type)] = poly
    return _wkt_data


def load_image(im_id: str) -> np.ndarray:
    return tiff.imread('./three_band/{}.tif'.format(im_id)).transpose([1, 2, 0])


def load_polygons(im_id: str, im_size: Tuple[int, int])\
        -> Dict[int, MultiPolygon]:
    w, h = im_size
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))

    x_max, y_min = get_x_max_y_min(im_id)
    x_scaler = w_ / x_max
    y_scaler = h_ / y_min

    def scale(polygons):
        return shapely.affinity.scale(
            polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    return {int(poly_type): scale(shapely.wkt.loads(poly))
            for poly_type, poly in get_wkt_data()[im_id].items()}


def mask_for_polygons(
        im_size: Tuple[int, int], polygons: MultiPolygon) -> np.ndarray:
    """ Return numpy mask for given polygons.
    polygons should already be converted to image coordinates.
    """
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def scale_percentile(matrix: np.ndarray) -> np.ndarray:
    """ Fixes the pixel value range to 2%-98% original distribution of values.
    """
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins

    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
