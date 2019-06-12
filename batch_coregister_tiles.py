#!/usr/bin/env python
from __future__ import print_function, division
import errno
import sys
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
from glob import glob
import argparse
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import gdal
import numpy as np
import geopandas as gpd
import pyproj
from shapely.geometry.polygon import Polygon
from mmaster_tools import reproject_geometry
from shapely.strtree import STRtree
from pybob.coreg_tools import dem_coregistration
from pybob.GeoImg import GeoImg


def clean_coreg_dir(out_dir):
    search_str = glob(os.path.sep.join([out_dir, 'tmp*.tif']))
    for f in search_str:
        os.remove(f)


def mkdir_p(outdir):
    try:
        os.makedirs(outdir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(outdir):
            pass
        else:
            raise


def buffer_conversion(fprint, buff_orig):
    a = 6378137
    e2 = 0.00669437999014
    
    lat = fprint.centroid.y
    my_lat_m = (np.pi * a * (1 - e2)) / (180 * (1 - e2 * np.sin(np.pi/180 * lat)**2)**1.5)
    my_lon_m = (np.pi * a * np.cos(np.pi/180 * lat)) / (180 * np.sqrt(1 - e2 * np.sin(np.pi/180 * lat)**2))
    
    return 1 / np.sqrt(my_lat_m**2 + my_lon_m**2) * buff_orig


def get_tiles(img, master_tiles, s):
    dname = os.path.splitext(os.path.basename(img))[0]
    tmp = GeoImg(img)

    my_proj = pyproj.Proj(master_tiles.crs)
    fprint = Polygon(tmp.find_corners(mode='xy'))
    fprint = reproject_geometry(fprint, tmp.spatialReference.ExportToProj4(), master_tiles.crs)

    if my_proj.is_latlong():
        buff = buffer_conversion(fprint, 1000)
    else:
        unit = [st.split('=')[-1] for st in my_proj.srs.split(' ') if 'units' in st]
        if unit[0] == 'm':  # have to figure out what to do with non-meter units...
            buff = 1000


    res = s.query(fprint.buffer(1000))
    # get only the results that intersect our buffered footprint
    intersects = [c for c in res if fprint.buffer(buff).intersection(c).area > 0]
    fnames = [master_tiles['filename'][master_tiles['geometry'] == c].values[0] for c in intersects]
    paths = [master_tiles['path'][master_tiles['geometry'] == c].values[0] for c in intersects]
    #subfolders = [master_tiles['subfolder'][master_tiles['geometry'] == c].values[0] for c in intersects]
    
    #tilelist = [os.path.sep.join([paths[i], subfolders[i], f]) for i, f in enumerate(fnames)]
    tilelist = [os.path.sep.join([paths[i], f]) for i, f in enumerate(fnames)]
    # create a temporary VRT from the tiles
    gdal.BuildVRT('tmp_{}.vrt'.format(dname), tilelist, resampleAlg='bilinear')
    # print(os.path.sep.join([os.path.abspath(indir), 'tmp_{}.vrt'.format(dname)]))
    return 'tmp_{}.vrt'.format(dname)


def batch_wrapper(argsin):
    arg_dict, master_tiles, s = argsin
    arg_dict['masterDEM'] = get_tiles(arg_dict['slaveDEM'], master_tiles, s)
    print(arg_dict['slaveDEM'])
    mkdir_p(arg_dict['outdir'])
    logfile = open(os.path.join(arg_dict['outdir'], 'coreg_' + str(os.getpid()) + '.log'), 'w')
    errfile = open(os.path.join(arg_dict['outdir'], 'coreg_' + str(os.getpid()) + '_error.log'), 'w')

    sys.stdout = logfile
    sys.stderr = errfile
    
    dem_coregistration(**arg_dict)
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    logfile.close()
    errfile.close()
    clean_coreg_dir(arg_dict['outdir'])


def main():
    np.seterr(all='ignore')
    # add master, slave, masks to argparse
    # can also add output directory
    parser = argparse.ArgumentParser(description="Iteratively calculate co-registration \
                                     parameters for two DEMs, as seen in Nuth and Kaeaeb (2011).")
    parser.add_argument('master_tiles', type=str, help='path to master DEM tiles to be used for co-registration')
    parser.add_argument('infiles', type=str, nargs='+', help='path to slave DEM(s) to be co-registered')
    parser.add_argument('-a', '--exc_mask', type=str, default=None,
                        help='Glacier mask. Areas inside of this shapefile will not be used for coregistration [None]')
    parser.add_argument('-b', '--inc_mask', type=str, default=None,
                        help='Land mask. Areas outside of this mask (i.e., water) \
                             will not be used for coregistration. [None]')
    parser.add_argument('-n', '--nproc', type=int, default=1, help='number of processors to use [1].')
    parser.add_argument('-p', '--points', action='store_true', default=False,
                        help="Process assuming that master DEM is ICESat data [False].")
    parser.add_argument('-f', '--full_ext', action='store_true', default=False,
                        help="Write full extent of master DEM and shifted slave DEM. [False].")
    args = parser.parse_args()

    print('Number of images given: {}'.format(len(args.infiles)))

    # open master_tiles, set up a search tree
    master_tiles = gpd.read_file(args.master_tiles)
    s = STRtree([f for f in master_tiles['geometry'].values])
    
    # only go through the trouble of setting up multiprocessing
    # if we have more than one directory to work on.
    if args.nproc > 1 and len(args.infiles) > 1:
        if args.nproc > mp.cpu_count():
            print("{} cores specified to use, but I could only find \
                  {} cores on this machine, so I'll use those.'".format(args.nproc, mp.cpu_count()))
            args.nproc = mp.cpu_count
        pool = mp.Pool(args.nproc, maxtasksperchild=1)
        # get a dictionary of arguments for each of the different DEMs,
        # starting with the common arguments (master dem, glacier mask, etc.)
        arg_dict = {'glaciermask': args.exc_mask, 
                    'landmask': args.inc_mask, 
                    'pts': args.points,
                    'full_ext': args.full_ext,
                    'return_var': False}
        u_args = [{'outdir': 'coreg_{}'.format(os.path.splitext(os.path.basename(d))[0]),
                   'slaveDEM': d} for i, d in enumerate(args.infiles)]
        for d in u_args:
            d.update(arg_dict)
        
        pool_args = [(u, master_tiles, s) for u in u_args]
        pool.map(batch_wrapper, pool_args, chunksize=1)
        pool.close()
        pool.join()
    else:
        odir = os.getcwd()
        for i, infile in enumerate(args.infiles):
            mst_dem = get_tiles(infile, master_tiles, s)
            outdir = 'coreg_{}'.format(os.path.splitext(os.path.basename(infile))[0])
            print('Running bias correction on {}'.format(infile))
            dem_coregistration(mst_dem,
                               infile,
                               glaciermask=args.exc_mask,
                               landmask=args.inc_mask,
                               outdir=outdir,
                               pts=args.points)
            os.chdir(odir)
    
    # clean up after ourselves, remove the vrts we created.
    for d in args.infiles:
        os.remove('tmp_{}.vrt'.format(os.path.splitext(os.path.basename(d))[0]))

    
if __name__ == "__main__":
    main()
