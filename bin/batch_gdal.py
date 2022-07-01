import argparse
import logging 
from pathlib import Path
import subprocess
from subprocess import PIPE
from typing import List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def run_subprocess(command, shell: bool = True):
    logger.debug(f'Run subprocess: {command})')
    proc = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=shell)
    proc_out = []
    for line in iter(proc.stdout.readline, b''):
        logger.debug('(subprocess) {}'.format(line.decode()))
        proc_out.append(line.decode().strip())
    proc_err = []
    for line in iter(proc.stderr.readline, b''):
        proc_err.append(line.decode().strip())
    if proc_err:
        logger.debug('(subprocess) {}'.format(proc_err))
    return proc_out, proc_err


def get_files(source_dirs: list, ext: str) -> List[Path]:
    # Get files matching extension
    files = []
    for sd in source_dirs:
        sd_files = list(Path(sd).rglob(f'*{ext}'))
        files.extend(sd_files)
    logger.info(f'Files found: {len(files)}')
    return files


def compute_statistics(files: List[Path], dryrun: bool = False):
    for in_raster in (pbar := tqdm(files)):
        # Compute statistics
        stats_cmd = f'gdalinfo -stats {str(in_raster)}'
        pbar.set_description(f'Calculatings stats - {in_raster.name}')
        if not dryrun:
            run_subprocess(stats_cmd)


def build_overviews(files: List[Path], dryrun=False):
    '''Create internal overviews'''
    for in_raster in (pbar := tqdm(files)):
        overviews_cmd = f'gdaladdo -r nearest {str(in_raster)}'
        pbar.set_description(f'Creating overviews - {in_raster.name}')
        if not dryrun:
            run_subprocess(overviews_cmd)
    

def create_tiles(files: List[Path], 
                 out_dir: Path, 
                 compress_method: Optional[str] = "LZW", 
                 dryrun=False):
    '''Create tiles optionally with compression'''
    for in_raster in (pbar := tqdm(files)):
        pbar.set_description(f'Creating overviews - {in_raster.name}')
        out_raster = out_dir / in_raster.name
        tile_cmd = f"gdal_translate {str(in_raster)} {out_raster} -co TILED=YES -co COPY_SRC_OVERVIEWS=YES"
        if compress_method is not None:
            tile_cmd += f" -co COMPRESS={compress_method}"
        if not dryrun:
            run_subprocess(tile_cmd)

    
def batch_gdal(source_dirs: List[str],
               ext: str,
               stats: bool = True,
               overviews: bool = True,
               tiles: bool = True,
               out_tiles_dir: Optional[str] = None,
               dryrun: bool = False):
    files = get_files(source_dirs, ext)
    if stats:
        compute_statistics(files, dryrun=dryrun)
    if overviews:
        build_overviews(files, dryrun=dryrun)
    if tiles:
        if not out_tiles_dir:
            raise argparse.ArgumentError('Must provide out_tiles_dir when creating tiles.')
        create_tiles(files, out_dir=out_tiles_dir, dryrun=dryrun)
    logger.info('Done.')    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source_dir', nargs='+', 
                        help='Source directories to search for files in, can specify more than one.')
    parser.add_argument('--ext')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--overviews', action='store_true')
    parser.add_argument('--tiles', action='store_true')
    parser.add_argument('--out_tiles_dir',
                        help='Path to write tiled rasters to - cannot be created in place.')
    parser.add_argument('--dryrun', action='store_true')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.tiles and args.out_tiles_dir is None:
        parser.error("--tiles requires --out_tiles_dir")
    
    batch_gdal(source_dirs=args.source_dir,
               ext=args.ext,
               stats=args.stats,
               overviews=args.overviews,
               tiles=args.tiles,
               out_tiles_dir=args.out_tiles_dir,
               dryrun=args.dryrun)
