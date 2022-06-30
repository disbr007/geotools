import argparse
import logging 
from pathlib import Path
import subprocess
from subprocess import PIPE
from typing import List

from tqdm import tqdm

# Args
source_dir = '/home/jeff/ao/data/proprietary/first_street_flood/tifs/1in100/2020'
ext = '.tif'
dryrun = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def run_subprocess(command, shell: bool = True):
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
    # output, error = proc.communicate()
    # logger.debug('Output: {}'.format(output.decode()))
    # logger.debug('Err: {}'.format(error.decode()))
    return proc_out, proc_err


def get_files(source_dirs: list, ext) -> List[Path]:
    # Get files matching extension
    files = []
    for sd in source_dirs:
        sd_files = list(Path(sd).rglob(f'*{ext}'))
        files.append(sd_files)
    logger.info(f'Files found: {len(files)}')
    return files


def build_overviews(source_dirs, ext, stats=True, overviews=True, dryrun=False):
    files = get_files(source_dirs, ext)
    processes_per_file = sum([1 for each in [stats, overviews] if each is True])
    with tqdm(total=len(files)*processes_per_file) as pbar:
        for f in files:
            if stats:
                # Compute statistics
                stats_cmd = f'gdalinfo -stats {str(f)}'
                pbar.set_description(f'Processing {f.name} (statistics)')
                if not dryrun:
                    run_subprocess(stats_cmd)
                pbar.update(1)
            if overviews:
                # Create internal overviews
                overviews_cmd = f'gdaladdo -r nearest {str(f)}'
                pbar.set_description(f'Processing {f.name} (overviews) ')
                if not dryrun:
                    run_subprocess(overviews_cmd)
                pbar.update(1)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--source_dir', nargs='+')
    parser.add_argument('--ext')
    parser.add_argument('--stats')
    parser.add_argument('--overviews')
    parser.add_argument('--dryrun', action='store_true')
    
    args = parser.parse_args()
    
    build_overviews(source_dir=args.source_dir,
                    ext=args.ext,
                    dryrun=args.dryrun)
    logger.info('Done.')
