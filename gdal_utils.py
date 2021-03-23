"""
Reproject a shapfile -- copied directly from ogr-cookbook, coverted to function
with in memory writing ability.
"""

import copy
import os
import glob
import logging
import posixpath
from pathlib import Path, PurePath
import subprocess
from subprocess import PIPE
import typing

from osgeo import gdal, ogr, osr

# from misc_utils.get_creds import get_creds
# from misc_utils.logging_utils import create_logger


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

ogr.UseExceptions()
gdal.UseExceptions()


def ogr_reproject(input_shp, to_sr, output_shp=None, in_mem=False):
    """
    Reproject shapefile using OGR.
    ** in memory reprojection not currently working /vsimem/**
    ** only works for polygons --> output geom_type needs to be fixed **
    """
    driver = auto_detect_ogr_driver(input_shp)

    # TODO: Improve the logic around in Memory Layers (not functional)
    if driver.GetName() == 'Memory' and isinstance(input_shp, ogr.DataSource):
        in_mem = True
        # If driver is Memory assume and ogr.DataSource is being passed
        # as input_shp
        # if isinstance(input_shp, ogr.DataSource):
        input_shp_name = 'mem_lyr'
        inLayer = input_shp.GetLayer(0)
        # # If driver is Memory and str: input_shp like /vsimem/vector.shp
        # elif isinstance(input_shp, str):
        #     # TODO: figure out how to handle viamem
        #     pass
    else:
        # if 'vsimem' in input_shp:
        #     in_mem = True

        # Get the input layer
        # Get names of from input shapefile path for output shape
        input_shp_name = os.path.basename(input_shp)
        input_shp_dir = os.path.dirname(input_shp)
        inDataSet = driver.Open(input_shp)
        inLayer = inDataSet.GetLayer(0)

    # Get the source spatial reference
    inSpatialRef = inLayer.GetSpatialRef()
    logger.debug('Input spatial reference: {}'.format(inSpatialRef.ExportToWkt()))
    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, to_sr)

    # Create the output layer
    # Default output shapefile name and location -- same as input
    if output_shp is None and in_mem is False:
        # TODO: Fix this
        output_shp = os.path.join(input_shp_dir, input_shp_name)
    # In memory output
    elif in_mem is True:
        output_shp = os.path.join('/vsimem', 'mem_lyr1.shp'.format(input_shp_name))
        # Convert windows path to unix path (required for gdal in-memory)
        output_shp = output_shp.replace(os.sep, posixpath.sep)

    # Check if output exists
    if os.path.exists(output_shp):
        logger.debug('Removing existing file: {}'.format(output_shp))
        remove_shp(output_shp)
    if in_mem is True:
        outDataSet = driver.CreateDataSource(os.path.basename(output_shp).split('.')[0])
    else:
        outDataSet = driver.CreateDataSource(os.path.dirname(output_shp))
    # TODO: Support non-polygon input types
    output_shp_name = os.path.basename(output_shp).split('.')[0]
    outLayer = outDataSet.CreateLayer(output_shp_name, geom_type=ogr.wkbMultiPolygon)

    # Add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # Get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()

    if in_mem is False and 'vsimem' not in input_shp:
        # Create .prj file
        outdir = os.path.dirname(output_shp)
        outname = os.path.basename(output_shp).split('.')[0]
        out_prj = os.path.join(outdir, '{}.prj'.format(outname))
        to_sr.MorphToESRI()

        file = open(out_prj, 'w')
        file.write(to_sr.ExportToWkt())
        file.close()

    # logger.debug('Output spatial reference: {}'.format(outLayer.GetSpatialRef().ExportToWkt()))
    # Save and close the shapefiles
    inLayer = None
    inDataSet = None
    outLayer = None
    outDataSet = None

    logger.debug('Projected ogr file: {}'.format(output_shp))

    return output_shp


def get_shp_sr(in_shp):
    """
    Get the crs of in_shp.
    in_shp: path to shapefile
    """
    driver = auto_detect_ogr_driver(in_shp)
    if driver.GetName() == 'Memory':
        # Memory driver doesn't support reading in memory datasets, (weird)
        # so use ESRI Shapefile which can read an in-memory datasets of
        # the form /vsimem/temp.shp
        driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(in_shp)
    lyr = ds.GetLayer()
    srs = lyr.GetSpatialRef()
    lyr = None
    ds = None
    return srs


def get_raster_sr(raster):
    """
    Get the crs of raster.
    raster: path to raster.
    """
    ds = gdal.Open(raster)
    prj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    prj = None
    ds = None
    return srs


def check_sr(shp_p, raster_p):
    """
    Check that spatial reference of shp and raster are the same.
    Optionally reproject in memory.
    """
     # Check for common spatial reference between shapefile and first raster
    shp_sr = get_shp_sr(shp_p)
    raster_sr = get_raster_sr(raster_p)

    if not shp_sr.IsSame(raster_sr):
        sr_match = False
        logger.debug('''Spatial references do not match...''')
        logger.debug('Shape SR: \n{} \nRaster SR:\n{}'.format(shp_sr, raster_sr))
    else:
        sr_match = True

    return sr_match


def auto_detect_ogr_driver(ogr_ds, name_only=False):
    """
    ***DEPRECIATED -- USE detect_ogr_driver()
    Autodetect the appropriate driver for an OGR datasource.

    Parameters
    ----------
    ogr_ds : OGR datasource
        Path to OGR datasource.

    Returns
    -------
    OGR driver OR
    OGR driver, layer name

    """
    logger.warning('auto_detect_ogr_driver() depreciated, use detect_ogr_driver()')
    # OGR driver lookup table
    GPKG = '.gpkg'
    driver_lut = {'.geojson': 'GeoJSON',
                  '.shp' : 'ESRI Shapefile',
                  GPKG: 'GPKG'
                  # TODO: Add more
                  }

    layer = None
    # Check if in-memory datasource
    if isinstance(ogr_ds, ogr.DataSource):
        driver_name = 'Memory'
    elif 'vsimem' in ogr_ds:
        # driver_name = 'Memory'
        driver_name = 'ESRI Shapefile'
    else:
        # Check if extension in look up table
        try:
            if GPKG in ogr_ds:
                ext = GPKG
                layer = ogr_ds.name
            else:
                ext = Path(ogr_ds).suffix

            if ext in driver_lut.keys():
                driver_name = driver_lut[ext]
            else:
                logger.info("""Unsupported driver extension {}
                                Defaulting to 'ESRI Shapefile'""".format(ext))
                driver_name = driver_lut['shp']
        except:
            logger.info('Unable to locate OGR driver for {}'.format(ogr_ds))
            driver_name = None

    if name_only:
        if layer:
            return driver_name, layer
        else:
            return driver_name
    else:
        try:
            driver = ogr.GetDriverByName(driver_name)
        except ValueError as e:
           print('ValueError with driver_name: {}'.format(driver_name))
           print('OGR DS: {}'.format(ogr_ds))
           raise e

        logger.debug('Driver autodetected: {}'.format(driver_name))

        return driver


def detect_ogr_driver(ogr_ds: str, name_only: bool = False) -> typing.Tuple[gdal.Driver, str]:
    """
    Autodetect the appropriate driver for an OGR datasource.

    Parameters
    ----------
    ogr_ds : OGR datasource
        Path to OGR datasource.
    name_only : bool
        True to return the name of the driver, else the ogr.Driver object will
        be return
    Returns
    -------
    OGR driver OR
    OGR driver, layer name

    """
    # TODO: convert strings to constants
    # Driver names
    FileGDB = 'FileGDB'
    OpenFileGDB = 'OpenFileGDB'
    # Suffixes
    GPKG = '.gpkg'
    SHP = '.shp'
    GEOJSON = '.geojson'
    GDB = '.gdb'

    supported_drivers = [gdal.GetDriver(i).GetDescription()
                         for i in range(gdal.GetDriverCount())]
    if FileGDB in supported_drivers:
        gdb_driver = FileGDB
    else:
        gdb_driver = OpenFileGDB

    # OGR driver lookup table
    driver_lut = {
        GEOJSON: 'GeoJSON',
        SHP: 'ESRI Shapefile',
        GPKG: 'GPKG',
        GDB: gdb_driver
                  }
    layer = None

    # Check if in-memory datasource
    if isinstance(ogr_ds, PurePath):
        ogr_ds = str(ogr_ds)
    if isinstance(ogr_ds, ogr.DataSource):
        driver = 'Memory'
    elif 'vsimem' in ogr_ds:
        driver = 'ESRI Shapefile'
    else:
        # Check if extension in look up table
        if GPKG in ogr_ds:
            drv_sfx = GPKG
            layer = Path(ogr_ds).name
        elif GDB in ogr_ds:
            drv_sfx = GDB
            layer = Path(ogr_ds).name
        else:
            drv_sfx = Path(ogr_ds).suffix

        if drv_sfx in driver_lut.keys():
            driver = driver_lut[drv_sfx]
        else:
            logger.warning("Unsupported driver extension {} "
                           "Defaulting to 'ESRI Shapefile'".format(drv_sfx))
            driver = driver_lut[SHP]

    logger.debug('Driver autodetected: {}'.format(driver))

    if not name_only:
        try:
            driver = ogr.GetDriverByName(driver)
        except ValueError as e:
           logger.error('ValueError with driver_name: {}'.format(driver))
           logger.error('OGR DS: {}'.format(ogr_ds))
           raise e

    return driver, layer


def remove_shp(shp):
    """
    Remove the passed shp path and all meta-data files.
    ogr.Driver.DeleteDataSource() was not removing
    meta-data files.

    Parameters
    ----------
    shp : os.path.abspath
        Path to shapefile to remove

    Returns
    ----------
    None

    """
    if shp:
        if os.path.exists(shp):
            logger.debug('Removing shp: {}'.format(shp))
            for ext in ['prj', 'dbf', 'shx', 'cpg', 'sbn', 'sbx']:
                meta_file = shp.replace('shp', ext)
                if os.path.exists(meta_file):
                    logger.debug('Removing metadata file: {}'.format(meta_file))
                    os.remove(meta_file)
            os.remove(shp)


def raster_bounds(path):
    '''
    GDAL only version of getting bounds for a single raster.
    '''
    src = gdal.Open(path)
    gt = src.GetGeoTransform()
    ulx = gt[0]
    uly = gt[3]
    lrx = ulx + (gt[1] * src.RasterXSize)
    lry = uly + (gt[5] * src.RasterYSize)

    return ulx, lry, lrx, uly


def minimum_bounding_box(rasters):
    '''
    Takes a list of DEMs (or rasters) and returns the minimum bounding box of all in
    the order of bounds specified for gdal.Translate.
    dems: list of dems
    '''
    ## Determine minimum bounding box
    ulxs, lrys, lrxs, ulys = list(), list(), list(), list()
    #geoms = list()
    for raster_p in rasters:
        ulx, lry, lrx, uly = raster_bounds(raster_p)
    #    geom_pts = [(ulx, lry), (lrx, lry), (lrx, uly), (ulx, uly)]
    #    geom = Polygon(geom_pts)
    #    geoms.append(geom)
        ulxs.append(ulx)
        lrys.append(lry)
        lrxs.append(lrx)
        ulys.append(uly)

    ## Find the smallest extent of all bounding box corners
    ulx = max(ulxs)
    uly = min(ulys)
    lrx = min(lrxs)
    lry = max(lrys)

    projWin = [ulx, uly, lrx, lry]

    return projWin


def clip_minbb(rasters, in_mem=False, out_dir=None, out_suffix='_clip',
               out_format='tif'):
    '''
    Takes a list of rasters and translates (clips) them to the minimum bounding box.

    Returns
    --------
    LIST : list of paths to the clipped rasters.
    '''
    projWin = minimum_bounding_box(rasters)
    logger.debug('Minimum bounding box: {}'.format(projWin))

    #  Clip to minimum bounding box
    translated = []
    for raster_p in rasters:
        if not out_dir and in_mem == False:
            out_dir = os.path.dirname(raster_p)
        elif not out_dir and in_mem==True:
            out_dir = '/vsimem'

        logging.info('Clipping {}...'.format(raster_p))
        if not out_suffix:
            out_suffix = ''

        raster_name = os.path.basename(raster_p).split('.')[0]

        raster_out_name = '{}{}.{}'.format(raster_name,
                                           out_suffix,
                                           out_format)

        raster_op = os.path.join(out_dir, raster_out_name)

        raster_ds = gdal.Open(raster_p)
        output = gdal.Translate(raster_op, raster_ds, projWin=projWin)
        if output is not None:
            translated.append(raster_op)
        else:
            logger.warning('Unable to translate raster: {}'.format(raster_p))

    return translated


def gdal_polygonize(img, out_vec, band=1, fieldname='label', overwrite=True):
    """
    Polygonize the band specified of the provided image
    to the out_vec vector file

    Parameters
    ----------
    img : os.path.abspath
        The raster file to be vectorized.
    out_vec : os.path.abspath
        The vector file to create.
    band : int, optional
        The raster band to vectorize. The default is 1.
    fieldname : str, optional
        The name of the field to create in the vector. The default is 'label'.
    overwrite : bool, optional
        True to overwrite if out_vec exists. The default is True.

    Returns
    -------
    status : int
        GDAL exit code: -1 indicates failure and triggers a logging message.

    """
    logger.info('Vectorizing raster:\n{}\n-->\n{}\n'.format(img, out_vec))
    if os.path.exists(out_vec) and overwrite:
        vec_base = '{}'.format(os.path.splitext(out_vec)[0])
        vec_meta_ext = ['dbf', 'shx', 'prj']
        vec_files = ['{}.{}'.format(vec_base, m) for m in vec_meta_ext
                      if os.path.exists('{}.{}'.format(vec_base, m))]
        vec_files.append(out_vec)
        logger.debug('Removing existing vector files: {}'.format('\n'.join(vec_files)))
        _del_files = [os.remove(f) for f in vec_files]

    # Open raster, get band and coordinate reference
    logger.debug('Opening raster for vectorization: {}'.format(img))
    # with gdal.Open(img) as src_ds:
    src_ds = gdal.Open(img)
    src_band = src_ds.GetRasterBand(band)

    src_srs = get_raster_sr(img)

    logger.debug('Raster SRS: {}'.format(src_srs.GetAuthorityCode(src_srs.ExportToWkt())))

    # Create vector
    logger.debug('Creating vector layer: {}'.format(out_vec))
    dst_driver, lyr_name = detect_ogr_driver(out_vec)
    if lyr_name is not None:
        db = str(Path(out_vec).parent)
        if not Path(db).exists():
            dst_ds = dst_driver.CreateDataSource(db)
        else:
            dst_ds = ogr.Open(db, update=True) # try opening without update
            existing_lyrs = [dst_ds.GetLayer(i).GetName()
                             for i in range(dst_ds.GetLayerCount())]
            if lyr_name in existing_lyrs:
                logger.debug('Removing existing layer: {}'.format(Path(db) / lyr_name))
                dst_ds.DeleteLayer(lyr_name)

    else:
        dst_ds = dst_driver.CreateDataSource(out_vec)
        # Drop extension for layer name
        lyr_name = os.path.basename(os.path.splitext(out_vec)[0])
    # if overwrite:
    #     options = ['OVERWRITE=YES']
    # else:
    #     options = None
    dst_lyr = dst_ds.CreateLayer(lyr_name, srs=src_srs)

    logger.debug('Vector layer SRS: {}'.format(dst_lyr.GetSpatialRef().ExportToWkt()))
    field_dfn = ogr.FieldDefn(fieldname, ogr.OFTString)
    dst_lyr.CreateField(field_dfn)

    # Polygonize
    logger.debug('Vectorizing...')
    status = gdal.Polygonize(src_band, None, dst_lyr, 0, [], callback=None)

    if status == -1:
        logger.error('Error during vectorization.')
        logger.error('GDAL exit code: {}'.format(status))

    src_ds = None

    return status


def match_pixel_size(rasters, dst_dir=None, sfx=None, resampleAlg='cubic', in_mem=False):

    rasters = [Path(r) for r in rasters]
    rasters_res = {}
    for r in rasters:
        src = gdal.Open(str(r))
        gt = src.GetGeoTransform()
        rasters_res[r] = (gt[1], gt[5])
        src1 = None

    max_x_raster = max(rasters_res.keys(), key=lambda k: abs(rasters_res[k][0]))
    max_y_raster = max(rasters_res.keys(), key=lambda k: abs(rasters_res[k][1]))
    outputs = [max_x_raster]
    if max_x_raster != max_y_raster:
        logger.error('Could not locate a raster with both maximum x-resolution and y-resolution.')
        print(rasters_res)
    else:
        logger.info('Maximum pixel size raster located: {}'.format(max_x_raster))
        rasters.remove(max_x_raster)
        max_x = rasters_res[max_x_raster][0]
        max_y = rasters_res[max_y_raster][1]
        logger.info('({}, {})'.format(max_x, max_y))
        for r in rasters:
            if in_mem == True:
                dst_dir = Path(r'/vsimem/')
            if not dst_dir:
                dst_dir = r.parent
            if not sfx:
                sfx = 'match_px_sz'
            dst = dst_dir / '{}_{}{}'.format(r.stem, sfx, r.suffix)
            if in_mem:
                dst = dst.as_posix()
            logger.info('Translating: {}'.format(r))
            logger.info('Destination: {}'.format(dst))
            trans_opts = gdal.TranslateOptions(xRes=max_x, yRes=max_y, resampleAlg=resampleAlg)
            output = gdal.Translate(destName=str(dst), srcDS=str(r), options=trans_opts)
            outputs.append(dst)

    return outputs


def get_raster_stats(raster, band_num=1):
    src = gdal.Open(raster)
    band = src.GetRasterBand(band_num)

    stats = band.GetStatistics(True, True)

    stats = {'min': stats[0],
             'max': stats[1],
             'mean': stats[2],
             'std': stats[3]}

    return stats


def run_subprocess(command):
    proc = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    for line in iter(proc.stdout.readline, b''):
        logger.info('(subprocess) {}'.format(line.decode()))
    proc_err = ""
    for line in iter(proc.stderr.readline, b''):
        proc_err += line.decode()
    if proc_err:
        logger.info('(subprocess) {}'.format(proc_err))
    output, error = proc.communicate()
    logger.debug('Output: {}'.format(output.decode()))
    logger.debug('Err: {}'.format(error.decode()))


def rescale_raster(raster, out_raster, out_min=0, out_max=1):
    logger.info('Determining input min/max...')
    stats = get_raster_stats(raster)
    logger.info('\nMin: {}\nMax:{}'.format(stats['min'], stats['max']))

    # cmd = "gdal_translate -scale {} {} {} {} {} {}".format(stats['min'], stats['max'],
    #                                                        out_min, out_max,
    #                                                        raster,
    #                                                        out_raster)
    # logger.debug('Running subprocess:\n{}'.format(cmd))
    # run_subprocess(cmd)
    # logger.info('Done.')
    logger.debug('Rescaling: {}'.format(raster))
    ds = gdal.Translate(out_raster, raster, scaleParams=[[stats['min'], stats['max'], out_min, out_max]])

    return ds


def stack_rasters(rasters, out, rescale=False, rescale_min=0, rescale_max=1):
    if rescale:
        rescaled = []
        for r in rasters:
            logger.info("Rescaling {}".format(r))
            rescaled_name = r'/vsimem/{}_rescale.vrt'.format(Path(r).stem)
            rescale_raster(str(r), rescaled_name, out_min=rescale_min, out_max=rescale_max)
            rescaled.append(rescaled_name)
        rasters = rescaled

    logger.info('Building stacked VRT...')
    temp = r'/vsimem/stack.vrt'
    gdal.BuildVRT(temp, rasters, separate=True)

    logger.info('Writing to: {}'.format(out))
    out_ds = gdal.Translate(out, temp)

    return out_ds


def rasterize_shp2raster_extent(ogr_ds, gdal_ds,
                                attribute=None,
                                burnValues=None,
                                where=None,
                                write_rasterized=False,
                                out_path=None,
                                nodata_val=None):
    """
    Rasterize a ogr datasource to the extent, projection, resolution of a given
    gdal datasource object. Optionally write out the rasterized product.
    ogr_ds           :    osgeo.ogr.DataSource OR os.path.abspath
    gdal_ds          :    osgeo.gdal.Dataset
    write_rasterised :    True to write rasterized product, must provide out_path
    out_path         :    Path to write rasterized product

    Writes
    Rasterized dataset to file.

    Returns
    osgeo.gdal.Dataset
    or
    None
    """
    logger.debug('Rasterizing OGR DataSource: {}'.format(ogr_ds))
    # If datasources are not open, open them
    if isinstance(ogr_ds, ogr.DataSource):
        pass
    else:
        ogr_ds = gdal.OpenEx(ogr_ds)

    if isinstance(gdal_ds, gdal.Dataset):
        pass
    else:
        gdal_ds = gdal.Open(gdal_ds)
    # Get DEM attributes
    if nodata_val is None:
        nodata_val = gdal_ds.GetRasterBand(1).GetNoDataValue()

    dem_sr = gdal_ds.GetProjection()
    dem_gt = gdal_ds.GetGeoTransform()
    x_min = dem_gt[0]
    y_max = dem_gt[3]
    x_res = dem_gt[1]
    y_res = dem_gt[5]
    x_sz = gdal_ds.RasterXSize
    y_sz = gdal_ds.RasterYSize
    x_max = x_min + x_res * x_sz
    y_min = y_max + y_res * y_sz
    gdal_ds = None

    ## Open shapefile
    # ogr_lyr = ogr_ds.GetLayer()

    # Create new raster in memory
    if write_rasterized is False:
        out_path = r'/vsimem/rasterized.tif'
        if os.path.exists(out_path):
            os.remove(out_path)

    ro = gdal.RasterizeOptions(format='GTiff', xRes=x_res, yRes=y_res,
                               width=x_sz, height=y_sz, attribute=attribute,
                               burnValues=burnValues,
                               where=where)

    # driver = gdal.GetDriverByName('GTiff')
    # out_ds = driver.Create(out_path, x_sz, y_sz, 1, gdal.GDT_Float32)
    # out_ds.SetGeoTransform((x_min, x_res, 0, y_min, 0, -y_res))
    # out_ds.SetProjection(dem_sr)
    # band = out_ds.GetRasterBand(1)
    # band.SetNoDataValue(
    #     dem_no_data_val)  # fix to get no_data_val before(?) clipping rasters
    # band.FlushCache()

    out_ds = gdal.Rasterize(out_path, ogr_ds, options=ro)
    out_ds.GetRasterBand(1).SetNoDataValue(nodata_val)
    # if write_rasterized is False:
    #     return out_ds
    # else:
    #     out_ds = None
    #     return out_path
    out_ds = None

    return out_path
