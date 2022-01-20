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
    gdf = gpd.read_file(geojson, drive='GeoJSON')
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

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--input_geojson', type=os.path.abspath)
    parser.add_argument('-og', '--out_geojson', type=os.path.abspath)
    
    parser.add_argument('-it', '--input_test_file', type=os.path.abspath)
    
    parser.add_argument('-wkt', '--input_wkt', type=str)
    parser.add_argument('-ow', '--out_wkt', type=os.path.abspath)
    
    args = parser.parse_args()
    
    main(args)
    