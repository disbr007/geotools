import argparse
import os

from gpd_utils import load_excel_points

    
def excel2geojson(excel_file, out_geojson=None, 
                  lat_col='Latitude',
                  lon_col='Longitude'):
    print(f'Loading points: {excel_file}')
    gdf = load_excel_points(excel_file=excel_file,
                            lat_col=lat_col,
                            lon_col=lon_col)
    print(f'{len(gdf):,} points loaded.')
    if out_geojson:
        print(f'Writing to file: {out_geojson}')
        gdf.to_file(out_geojson, driver='GeoJSON')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_excel', type=os.path.abspath)
    parser.add_argument('-og', '--out_geojson', type=os.path.abspath)
    
    parser.add_argument('--lat_col', default='Latitude',
                        help='Name of column with latitude.')
    parser.add_argument('--lon_col', default='Longitude',
                        help='Name of column with longitude.')
    
    args = parser.parse_args()
    
    excel2geojson(excel_file=args.input_excel,
                  out_geojson=args.out_geojson,
                  lat_col=args.lat_col,
                  lon_col=args.lon_col)
