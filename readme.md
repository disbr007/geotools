# geotools
Commonly used geospatial functions. Tools are written both at the 
OGR/GDAL level as well as using abstracted libraries like shapely,
geopandas, rasterio, etc.

### Set Up
Clone geotools repository:  
```
git clone https://github.com/disbr007/geotools.git
```
Install geotools:
```
pip install [path/to/geotools/]
```  
Test installation:
```
python
>>> import geotools
```

# Release process
Currently the release process is relatively manual.
1. Identify most recent tag:
`git tag`
2. Update `geotools/version.py` with the new version.
3. Push changes to `main`
4. Create new tag with appropriate bump in patch, minor, or major number, matching `geotools/version.py`:
`git tag v0.1.2`
5. Push tag to GitHub:
`git push origin v0.1.2`
6. GitHub Actions will create a release from any push of a version tag: [release.yml](.github/workflows/version.yml)
