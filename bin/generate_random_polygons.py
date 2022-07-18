import argparse
import logging
import os
from pathlib import Path

import fiona

from geotools.gpdtools import generate_random_polygons

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

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
    parser.add_argument('--epsg_processing',
                        help='The epsg to do processing, i.e. the epsg that specifieds '
                        'the units to interpret arguments in.')
    parser.add_argument('--epsg_out',
                        help='The epsg to write the result as.')
    parser.add_argument('--densify_percentage', type=float,
                        help='Percentage by which to densify polygon verticies before randomizing.'
                             'This is helpful if a very simple polygon is passed, e.g. a square.')
    parser.add_argument('--simplify', type=float,
                        help='Distance allowed to simplify polygon.')
    parser.add_argument('--use_centroids', action='store_true',
                        help='Seed random polygons with centroids of bounding_polygons. If '
                        'len(bounding_polygons) < n, random points will be added on top. If l'
                        'len(bounding_polygons) > n, a random selection of n centroids will be used.')
    
    # DEBUGGING
    # import sys, shlex, os
    # os.chdir(r'/home/jeff/ao/data/baa/debugging/parcels')
    # # # args_str = ('-i /home/jeff/data/gis/tiger_lines_2020/tl_2020_us_state/tl_2020_us_state.shp '
    # # #             '-o /home/jeff/ao/data/baa/debugging/random_500_polys_in_states.kml '
    # # #             '-n 250 -s 100 --epsg_processing 2163 --epsg_out 4326 -d 25 --size_fluff 75 '
    # # #             '--densify_percentage 250')
    # args_str = ('-i /home/jeff/data/gis/tiger_lines_2020/tl_2020_us_state/tl_2020_us_state_continental.shp '
    #             '-o /home/jeff/ao/data/baa/debugging/random_200_polys_in_states.kml '
    #             '-n 250 -s 500 --epsg_processing 3857 --epsg_out 4326 -d 10 '
    #             '--densify_percentage 50')
    # cli_args = shlex.split(args_str)
    # sys.argv = [__file__]
    # sys.argv.extend(cli_args)

    args = parser.parse_args()
    
    logger.info('Generating random polygons...')
    random_polys = generate_random_polygons(bounding_poly=args.bounding_polygons,
                                            n=args.number_polygons,
                                            randomize_distance=args.randomize_distance,
                                            size=args.size,
                                            size_fluff=args.size_fluff,
                                            epsg=args.epsg_processing,
                                            densify_percentage=args.densify_percentage,
                                            simplify_tolerance=args.simplify,
                                            start_with_centroids=args.use_centroids)

    if args.outfile:
        driver = None
        if Path(args.outfile).suffix.lower() == '.kml':
            if args.epsg_out and args.epsg_out != '4326':
                print(f'Warning, overriding --epsg_out ({args.epsg_out}) with '
                      'KML standard: "4326"')
            # Use 4326
            random_polys = random_polys.to_crs(epsg='4326')
            # KML driver can't replace file
            if Path(args.outfile).exists():
                print('Replacing existing file...')
                os.remove(args.outfile)
            driver = 'KML'
            # random_polys.to_file(args.outfile, driver=driver)
        if args.epsg_out:
            random_polys = random_polys.to_crs(epsg=args.epsg_out)
        random_polys.to_file(args.outfile, driver=driver)
        print(f'Writing to {args.outfile}')
        
    print('Done.')

        
        
