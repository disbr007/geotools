import geopandas as gpd
from shapely.geometry import box


box_str = '421492.05012084 127958.158773347,422296.759310653 128663.45960685'
crs = 27700
out_shp = r'/home/jeff/ao/data/baa/uk_baa/ancient_woodlands/test_feature_box.shp'

mins, maxs = box_str.split(',')
minx, miny = mins.split(' ')
maxx, maxy = maxs.split(' ')
box_geom = box(float(minx), float(miny), float(maxx), float(maxy))

gdf = gpd.GeoDataFrame(geometry=[box_geom], crs=f'epsg:{crs}')

gdf.to_file(out_shp)



