import argparse
import json
import os

import shapely
import geopandas as gpd


PARAMATER = 'parameter'
LOCATION = 'location'
WKT = 'wkt'

def wkt2geojson(wkt_str, geojson_out, crs='epsg:4326'):
    print(f'Converting wkt to geojson: {geojson_out}')
    print(wkt_str[0:100])
    geom = shapely.wkt.loads(wkt_str)
    gdf = gpd.GeoDataFrame({'id': [1],
                            'geometry': [geom]},
                           crs=crs)
    gdf.to_file(geojson_out, driver='GeoJSON')
    

def geojson2wkt(geojson, wkt_out = None):
    print(f'Converting {geojson} to wkt ({wkt_out})')
    gdf = gpd.read_file(geojson, driver='GeoJSON')
    wkts = gdf.geometry.to_wkt()
    if wkt_out:
        with open(wkt_out, 'w') as out:
            for w in wkts:
                out.write(w)
                out.write('\n')


def test_file2geojson(test_file, geojson_out):
    print(f'Converting test file to geojson: {test_file}')
    with open(test_file, 'r') as src:
        data = json.load(src)
    wkt = data[PARAMATER][LOCATION][WKT]
    wkt2geojson(wkt_str=wkt, geojson_out=geojson_out)


def aodrawing2geojson(drawing_file, out_dir):
    print(f'Converting AO Drawing file to geojson: {drawing_file}')
    with open(drawing_file, 'r') as src:
        data = json.load(src)
    HAZARDS = 'hazards'
    BUILDABLE = 'buildable'
    NAME = 'name'
    ITEMS = 'items'
    WKT = 'wkt'
    
    hazards = data[HAZARDS]
    for haz in hazards:
        if len(haz[ITEMS]) == 0:
            print(f'No records for {haz[NAME]}')
            continue
        else:
            out_name = os.path.join(out_dir, f'{haz[NAME]}.geojson')
            gdf = gpd.GeoDataFrame(haz[ITEMS])
            gdf['geometry'] = gdf[WKT].apply(lambda x: shapely.wkt.loads(x))
            print(f'Writing {haz[NAME]} to file: {out_name}')
            gdf.to_file(out_name, driver='GeoJSON')
    # Buildable
    buildable_gdf = gpd.GeoDataFrame(data[BUILDABLE][ITEMS])
    buildable_gdf['geometry'] = buildable_gdf[WKT].apply(lambda x: shapely.wkt.loads(x))
    out_name = os.path.join(out_dir, 'buildable.geojson')
    print(f'Writing buildable to: {out_name}')
    buildable_gdf.to_file(out_name, driver='GeoJSON')
    
    
def main(args):
    if args.input_wkt:
        if os.path.exists(args.input_wkt):
            print('Found path to WKT file.')
            with open(args.input_wkt, 'r') as src:
                input_wkt = src.read()
        else:
            input_wkt = args.input_wkt
        wkt2geojson(input_wkt, args.out_geojson)
    if args.input_geojson:
        geojson2wkt(args.input_geojson, args.out_wkt)
    if args.input_test_file:
        test_file2geojson(args.input_test_file, args.out_geojson)
    if args.input_drawing_file:
        out_dir = os.getcwd() if args.out_dir is None else args.out_dir
        aodrawing2geojson(drawing_file=args.input_drawing_file, out_dir=out_dir)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--input_geojson', type=os.path.abspath)
    parser.add_argument('-og', '--out_geojson', type=os.path.abspath)
    
    parser.add_argument('-it', '--input_test_file', type=os.path.abspath)
    
    parser.add_argument('-wkt', '--input_wkt', type=str)
    parser.add_argument('-ow', '--out_wkt', type=os.path.abspath)
    
    parser.add_argument('-df', '--input_drawing_file', type=os.path.abspath)
    parser.add_argument('--out_dir', type=os.path.abspath)
    
    
    # import sys, shlex
    # os.chdir(r'/home/jeff/ao_code/task-geoanalysis/tmp/work/geo/dash_mvb9gddudash_tfxsxzilbe')
    # args_str = ('-df drawing_zWAmx.json --out_dir /home/jeff/scratch')
    #             # '-ct substation')
    # cli_args = shlex.split(args_str)
    # sys.argv = [__file__]
    # sys.argv.extend(cli_args)
    
    args = parser.parse_args()
    
    main(args)
    