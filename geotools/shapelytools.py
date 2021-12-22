import operator
import random

import shapely
from shapely.geometry import Point, Polygon


def point2square(point: Point, size: float, size_fluff: float = None):
    x, y = point.x, point.y
    if size_fluff:
        size = random.uniform(size-size_fluff, size+size_fluff)
    half = size / 2
    ul = Point(x+half, y-half)
    ur = Point(x+half, y+half)
    lr = Point(x-half, y+half)
    ll = Point(x-half, y-half)
    poly = Polygon([ul, ur, lr, ll])
    return poly


def densify_polygon(polygon: Polygon, densify_percent: float):
    coords = len(polygon.exterior.coords)
    addtl_points = round(coords * (densify_percent/100))
    interval = polygon.exterior.length / addtl_points
    new_points = [polygon.exterior.interpolate(interval*i) for i in range(addtl_points)]
    new_poly = Polygon(new_points)
    dense_poly = shapely.ops.unary_union([polygon, new_poly])
    return dense_poly


def randomize_verticies(polygon: Polygon, max_distance: float):
    coords = polygon.exterior.coords
    ops = (operator.add, operator.sub)
    randomized_coords = [Point(random.choice(ops)(c[0], random.uniform(0, max_distance)),
                               random.choice(ops)(c[1], random.uniform(0, max_distance)))
                         for c in coords]
    randomized_poly = Polygon(randomized_coords)
    if not randomized_poly.is_valid:
        randomized_poly = randomized_poly.buffer(0)
    
    return randomized_poly
