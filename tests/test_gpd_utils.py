import json
import unittest
import pytest_check as check
import sys
import os

import geopandas as gpd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpd_utils import esri2gdf

ESRIJSON_EXAMPLE = r'/home/jeff/scratch/esrijson_example.json'


class TestEsri2GDF(unittest.TestCase):
    def test_esri2gdf(self):
        with open(ESRIJSON_EXAMPLE) as src:
            data = json.load(src)
            features = data['features']
        result = esri2gdf(features)
        check.is_instance(result, gpd.GeoDataFrame)