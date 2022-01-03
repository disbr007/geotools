import argparse
import os
from pathlib import Path

import fiona

from geotools.gpdtools import generate_random_polygons

fiona.supported_drivers['KML'] = 'rw'
fiona.supported_drivers['kml'] = 'rw'
fiona.supported_drivers['LIBKML'] = 'rw'
fiona.supported_drivers['libkml'] = 'rw'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--bounding_polygons', type=os.path.abspath,
                        help='Path to file containing polygon(s) to generate polygons within.')
    parser.add_argument('-o', '--outfile', type=os.path.abspath,
                        help='Path to write randomized polygons to.')
    parser.add_argument('-n', '--number_polygons', type=int,
                        help='Number of random polygons to generate')
    parser.add_argument('-d', '--randomize_distance', type=float,
                        help='Maximum distance to move a vertex')
    parser.add_argument('-s', '--size', type=float,
                        help='Size of polygons to generate')
    parser.add_argument('--size_fluff', type=float,
                        help='Allow randomization of sizes up to this value')
    parser.add_argument('--epsg')
    parser.add_argument('--densify_percentage', type=float,
                        help='Percentage by which to densify polygon verticies before randomizing.'
                             'This is helpful if a very simple polygon is passed, e.g. a square.')
    parser.add_argument('--simplify', type=float,
                        help='Distance allowed to simplify polygon.')
    
    # DEBUGGING
    # import sys, shlex, os
    # os.chdir(r'/home/jeff/ao/data/baa/debugging/2021dec23_flood')
    # args_str = ('-i states_subset.shp -o random_in_states.kml -n 1000 -d 50 -s 250 --size_fluff 240 --epsg 2163 --densify_percentage 300 --simplify 50')
    # cli_args = shlex.split(args_str)
    # sys.argv = [__file__]
    # sys.argv.extend(cli_args)

    args = parser.parse_args()
    
    random_polys = generate_random_polygons(bounding_poly = args.bounding_polygons,
                                            n=args.number_polygons,
                                            randomize_distance=args.randomize_distance,
                                            size=args.size,
                                            size_fluff=args.size_fluff,
                                            epsg=args.epsg,
                                            densify_percentage=args.densify_percentage,
                                            simplify_tolerance=args.simplify)

    if args.outfile:
        if Path(args.outfile).suffix.lower() == '.kml':
            # Use 4326
            random_polys = random_polys.to_crs(epsg='4326')
            # KML driver can't replace file
            if Path(args.outfile).exists():
                print('Replacing existing file...')
                os.remove(args.outfile)
            driver = 'KML'
            random_polys.to_file(args.outfile, driver=driver)
        random_polys.to_file(args.outfile)
        print(f'Writing to {args.outfile}')

        
        
