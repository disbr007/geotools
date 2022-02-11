import math
import operator
import random

import numpy as np
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


def create_grid_cell_size(cell_size, bounds):
    xmin, ymin, xmax, ymax = bounds
    xdist = xmax - xmin
    ydist = ymax - ymin

    num_x = math.ceil(xdist // cell_size) + 1
    num_y = math.ceil(ydist // cell_size) + 1

    grid_cells = []
    x0 = xmin
    y0 = ymin
    for x_ct in range(num_x):
        x1 = x0 + cell_size
        for y_ct in range(num_y):
            y1 = y0 + cell_size
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
            y0 = y1
        x0 = x1
        y0 = ymin

    return grid_cells


def create_grid_cell_count(n_cells, bounds):
    xmin, ymin, xmax, ymax = bounds
    
    cell_size = (xmax-xmin)/n_cells

    # create the cells in a loop with a one cell buffer above and below
    grid_cells = []
    # for x0 in tqdm(np.arange(xmin-cell_size, xmax+cell_size, cell_size)):
    #     for y0 in np.arange(ymin-cell_size, ymax+cell_size, cell_size):
    for x0 in np.arange(xmin, xmax, cell_size):
        for y0 in np.arange(ymin, ymax, cell_size):
            # bounds
            x1 = x0-cell_size
            y1 = y0+cell_size
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

    return grid_cells