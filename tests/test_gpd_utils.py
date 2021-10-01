import json
import unittest
from pathlib import Path
import pytest_check as check
import sys
import os

import geopandas as gpd

from geotools.gpdtools import esri2gdf

ESRIJSON_EXAMPLE = Path(__file__).parent / r'test_data/esrijson_example.json'


class TestEsri2GDF(unittest.TestCase):
    def test_esri2gdf(self):
        with open(ESRIJSON_EXAMPLE) as src:
            data = json.load(src)
            features = data['features']
        result = esri2gdf(features)
        check.is_instance(result, gpd.GeoDataFrame)