"""
Based largely on code written by Eric Anderson:
https://github.com/anderson-optimization/ao-drawing-spec/blob/master/kml_to_drawing.py
"""

import logging
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Union
import zipfile

from fastkml import kml, Folder, Placemark, LineStyle, PolyStyle, Document
import fastkml
import pandas as pd
import geopandas as gpd
import shapely.ops
from tqdm import tqdm


logger = logging.getLogger(__name__)



def _to_2d(x: float, y: float, z: float) -> tuple:
    return tuple(filter(None, [x, y]))


def loop_features(element: Union[kml.Document, kml.Placemark, kml.Folder],
                  folder_path: str) -> Generator:
    logger.debug(f'Iterating features in folder: {folder_path}')
    features = element.features()
    # for f in tqdm(features, desc='Iterating features'):
    for f in features:
        if isinstance(f, Folder):
            for item in loop_features(f, folder_path+f.name+'/'):
                yield item
        if isinstance(f, Placemark):
            yield {
                "f": f,
                "folder": folder_path
            }
        if isinstance(f, Document):
            for item in loop_features(f, folder_path+f.name+'/'):
                yield item


def get_style_obj(style: kml.Style) -> dict:
    # logger.debug('Havestyle', style)
    obj = {}
    if hasattr(style, 'styles'):
        for s in style.styles():
            try:
                if isinstance(s, LineStyle) and s.color is not None:
                    opacity = s.color[:2]
                    color = s.color[2:]
                    obj['strokeOpacity'] = opacity
                    obj['strokeColor'] = '#'+color.upper()
                    obj['strokeWeight'] = s.width        
                if isinstance(s, PolyStyle):
                    opacity = s.color[:2]
                    color = s.color[2:]
                    obj['fillColor'] = '#'+color.upper()
                    obj['fillOpacity'] = opacity
            except TypeError as e:
                logger.error('TypeError getting style')
                logger.error(e)
    return obj


def read_kmz(kmz: str) -> kml.KML:
    """
    Reads data from a KML within a KMZ

    Parameters
    ----------
    kmz: str
        The KMZ or KML file to work with

    Returns
    -------
    kml.KML
    """
    f = zipfile.ZipFile(kmz)
    # Confirm only one KML exists
    files = list(f.namelist())
    kml_files = [f for f in files if f.endswith('.kml')]
    # TODO: Handle multiple KML files
    # Exit if more than one KML file found
    if len(kml_files) > 1:
        logger.error(f'Multiple KML files not supported: {kml_files}')
        raise Exception    
    kml_file = kml_files[0]
    
    # Read KML data
    kml_data = f.open(kml_file).read()

    return kml_data


def read_kml(kml_file: str) -> kml.KML:
    # Read unzipped KML
    with open(kml_file, 'rb') as src:
        kml_data = src.read()
    if not kml_data:
        logger.error('No doc loaded')
        raise Exception
    
    return kml_data
    

def load_placemarks(kml_obj: kml.KML) -> Tuple[List[Dict], Dict]:
    # Load styles and features(?)
    styles = {}
    items = []
    for doc in kml_obj.features():
        logger.debug(f'Parsing KML doc: {doc.name}')
        for style in doc.styles():
            styles[style.id] = style

        new_items = loop_features(doc, '/')
        items += new_items

    return items, styles


def attributes_from_extended_data(extended_data: kml.ExtendedData) -> dict:
    attributes = {}
    for element in extended_data.elements:
        for d in element.data:
            attributes[d['name']] = d['value']
            if d['name'] in'Name':
                attributes['name'] = d['value']
    
    return attributes


def attributes_from_description(description: str) -> dict:
    """
    Parses attributes (key-value pairs) from an HTML table.

    Parameters
    ----------
    description: str
        String of HTML to parse

    Returns
    -------
    dict:
        Dictionary of key-value pairs
    """
    attributes = {}
    data = pd.read_html(description)
    if len(data) == 2:
    # try:
        table = data[1]
        for _idx, row in table.iterrows():
            key = row[0]
            value = row[1]
            if isinstance(value, float):
                continue
            if value:
                attributes[key] = value
    # except Exception as e:  # What exception(s) is being caught here?
    elif len(data) == 1:
        table = data[0]
        for _idx, row in table.iterrows():
            key = row[0]
            value = row[1]
            if key == value:
                continue
            if isinstance(value, float):
                continue
            if value:
                attributes[key] = value
    else:
        logger.error(f'Error reading table for placemark. Data: {data}')

    return attributes


def create_record(placemark: kml.Placemark) -> dict:
    geo = shapely.ops.transform(_to_2d, placemark.geometry)

    record = {
        "name": placemark.name,
        "wkt": geo.wkt,
        "geometry": geo
    }

    # Parse attributes from extended_data (SimpleField falls under this)
    if placemark.extended_data:
        attributes = attributes_from_extended_data(placemark.extended_data)
        record.update(attributes)
        
    # Parse attributes from description field
    if placemark.description:
        # Read html string into list of pd.DataFrame(s)
        attributes = attributes_from_description(placemark.description)
        record.update(attributes)
    
    return record


def capture_style(placemark: kml.Placemark, styles: dict) -> dict:
    style_url = placemark.styleUrl.replace('#', '')
    if style_url in styles.keys():
        # Style map, this should be handled better
        # logger.info(style_url)
        # if hasattr(style, 'normal'):
            # style_url = style.normal.url.replace('#', '')
            
        style = styles[style_url]
        style_obj = get_style_obj(style)
    else:
        style_obj = {}

    return style_obj


def kmz2gdf(kmz: str, parse_style: bool = False, include_folder_path: bool = False) -> gpd.GeoDataFrame:
    logger.info('Parsing KMZ...')

    # Load KML data
    if zipfile.is_zipfile(kmz):
        kml_data = read_kmz(kmz=kmz)
    else:
        kml_data = read_kml(kml_file=kmz)
    
    kml_obj = kml.KML()
    kml_obj.from_string(kml_data)

    placemarks, styles = load_placemarks(kml_obj)

    out_records = []
    for i, item in tqdm(enumerate(placemarks), desc='Processing KMZ features', 
                        total=len(placemarks)):
        placemark = item['f']
        record = create_record(placemark)

        if parse_style:
            record['style'] = capture_style(placemark, styles)
        if include_folder_path:
            record['folder_path'] = item['folder']
        out_records.append(record)
        
    # Create GeoDataFrame
    # Assumes input KMZ is in 4326
    gdf = gpd.GeoDataFrame(out_records, geometry='geometry', crs='epsg:4326')
    
    return gdf


def split_gdf_on_folder_path(gdf: gpd.GeoDataFrame, folder_path_field: str='folder_path') -> List[gpd.GeoDataFrame]:
    new_gdfs = {}
    unique_fps = gdf[folder_path_field].unique()
    for uf in unique_fps:
        subset = gdf[gdf[folder_path_field]==uf]
        new_gdfs[Path(uf).stem] = subset
    return new_gdfs
