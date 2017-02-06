import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import logging
import json
from typing import Dict, Tuple
import sys

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import shapely.geometry
import tifffile as tiff


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


def load_image(im_id: str, rgb_only=False, align=True) -> np.ndarray:
    im_rgb = tiff.imread('./three_band/{}.tif'.format(im_id)).transpose([1, 2, 0])
    if rgb_only:
        return im_rgb
    im_p = np.expand_dims(tiff.imread('sixteen_band/{}_P.tif'.format(im_id)), 2)
    im_m = tiff.imread('sixteen_band/{}_M.tif'.format(im_id)).transpose([1, 2, 0])
    im_a = tiff.imread('sixteen_band/{}_A.tif'.format(im_id)).transpose([1, 2, 0])
    h, w = im_rgb.shape[:2]
    if align:
        logger.info('Getting alignment')
        im_p = _aligned(im_rgb, im_p)
        im_m = _aligned(im_rgb, im_m)
        im_a = _aligned(im_rgb, im_a)
    if im_p.shape != im_rgb.shape[:2]:
        im_p = cv2.resize(im_p, (w, h), interpolation=cv2.INTER_CUBIC)
    im_p = np.expand_dims(im_p, 2)
    im_m = cv2.resize(im_m, (w, h), interpolation=cv2.INTER_CUBIC)
    im_a = cv2.resize(im_a, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.concatenate([im_rgb, im_p, im_m, im_a], axis=2)


def _preprocess_for_alignment(im):
    im = im.astype(np.float32)
    im = np.mean(im, 2)
    return _get_gradient(im)


def _get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    # Combine the two gradients
    grad = cv2.addWeighted(
        np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def _aligned(im_ref, im):
    w, h = im.shape[:2]
    im_ref = cv2.resize(im_ref, (w, h), interpolation=cv2.INTER_CUBIC)
    im_ref = _preprocess_for_alignment(im_ref)
    im_gray = _preprocess_for_alignment(im)
    try:
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-7)
        _, warp_matrix = cv2.findTransformECC(
            im_ref, im_gray, warp_matrix, warp_mode, criteria)
    except cv2.error as e:
        logger.info('Error getting alignment: {}'.format(e))
        return im
    else:
        logger.info('Got alignment: {}'
                    .format(str(warp_matrix).replace('\n', '')))
        im = cv2.warpAffine(
            im, warp_matrix, (im.shape[1], im.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        im[im == 0] = np.mean(im)
        return im


def load_polygons(im_id: str, im_size: Tuple[int, int])\
        -> Dict[int, MultiPolygon]:
    return {
        int(poly_type): scale_to_mask(im_id, im_size, shapely.wkt.loads(poly))
        for poly_type, poly in get_wkt_data()[im_id].items()}


def scale_to_mask(im_id: str, im_size: Tuple[int, int], poly: MultiPolygon)\
        -> MultiPolygon:
    x_scaler, y_scaler = get_scalers(im_id, im_size)
    return shapely.affinity.scale(
        poly, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


def dump_polygons(im_id: str, im_size: Tuple[int, int], polygons: MultiPolygon)\
        -> str:
    """ Save polygons for submission.
    """
    x_scaler, y_scaler = get_scalers(im_id, im_size)
    x_scaler = 1 / x_scaler
    y_scaler = 1 / y_scaler
    polygons = shapely.affinity.scale(
        polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    return shapely.wkt.dumps(polygons)


def get_scalers(im_id: str, im_size: Tuple[int, int]) -> Tuple[float, float]:
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    x_max, y_min = get_x_max_y_min(im_id)
    x_scaler = w_ / x_max
    y_scaler = h_ / y_min
    return x_scaler, y_scaler


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


def rotated(patch: np.ndarray, angle: float) -> np.ndarray:
    patch = patch.transpose([1, 2, 0]).astype(np.float32)
    size = patch.shape[:2]
    center = tuple(np.array(size) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    patch = cv2.warpAffine(patch, rot_mat, size, flags=cv2.INTER_LINEAR)
    if len(patch.shape) == 2:
        return np.expand_dims(patch, 0)
    else:
        return patch.transpose([2, 0, 1])


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


def mask_to_polygons(mask: np.ndarray, epsilon=5., min_area=10.,
                     fix=False) -> MultiPolygon:
    if fix:
        epsilon *= 2
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]
    if not contours:
        return MultiPolygon()

    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])

    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)

    all_polygons = MultiPolygon(all_polygons)
    all_polygons = shapely.wkt.loads(
        shapely.wkt.dumps(all_polygons, rounding_precision=8))
    if fix:
        all_polygons = all_polygons.buffer(-1)
    while not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
        all_polygons = shapely.wkt.loads(
            shapely.wkt.dumps(all_polygons, rounding_precision=8))
    return all_polygons


def load_mask_stats():
    with open('images/stats.json') as f:
        return json.load(f)


def mask_tp_fp_fn(pred_mask: np.ndarray, true_mask: np.ndarray,
                  threshold: float) -> Tuple[int, int, int]:
    pred_mask = pred_mask >= threshold
    true_mask = true_mask == 1
    return (( pred_mask &  true_mask).sum(),
            ( pred_mask & ~true_mask).sum(),
            (~pred_mask &  true_mask).sum())


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(module)s: %(message)s'))
    logger.addHandler(ch)
    return logger


logger = get_logger(__name__)


def imap_fixed_output_buffer(fn, it, threads: int):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        max_futures = threads + 1
        for x in it:
            while len(futures) >= max_futures:
                future, futures = futures[0], futures[1:]
                yield future.result()
            futures.append(executor.submit(fn, x))
        for future in futures:
            yield future.result()
