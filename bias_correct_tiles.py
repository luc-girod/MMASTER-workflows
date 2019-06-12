#!/usr/bin/env python
from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import argparse
import multiprocessing as mp
from glob import glob
import matplotlib
matplotlib.use('Agg')
import numpy as np
import geopandas as gpd
import gdal
import pyproj
from shapely.strtree import STRtree
from mmaster_tools import mmaster_bias_removal, get_aster_footprint


def batch_wrapper(argsin):
    arg_dict, master_tiles, s = argsin
    arg_dict['mst_dem'] = get_tiles(arg_dict['work_dir'], master_tiles, s)
    return mmaster_bias_removal(**arg_dict)


def buffer_conversion(fprint, buff_orig):
    a = 6378137
    e2 = 0.00669437999014
    
    lat = fprint.centroid.y
    my_lat_m = (np.pi * a * (1 - e2)) / (180 * (1 - e2 * np.sin(np.pi/180 * lat)**2)**1.5)
    my_lon_m = (np.pi * a * np.cos(np.pi/180 * lat)) / (180 * np.sqrt(1 - e2 * np.sin(np.pi/180 * lat)**2))
    
    return 1 / np.sqrt(my_lat_m**2 + my_lon_m**2) * buff_orig


def get_tiles(indir, master_tiles, s):
    dname = os.path.basename(indir.strip(os.path.sep))  # get the actual granule/folder name
    fprint = get_aster_footprint(indir, master_tiles.crs, indir=indir, polyout=False)
    my_proj = pyproj.Proj(master_tiles.crs)
    if my_proj.is_latlong():
        buff = buffer_conversion(fprint, 1000)
    else:
        unit = [st.split('=')[-1] for st in my_proj.srs.split(' ') if 'units' in st]
        if unit[0] == 'm':  # have to figure out what to do with non-meter units...
            buff = 1000
    res = s.query(fprint.buffer(buff))
    # get only the results that intersect our buffered footprint
    intersects = [c for c in res if fprint.buffer(buff).intersection(c).area > 0]
    fnames = [master_tiles['filename'][master_tiles['geometry'] == c].values[0] for c in intersects]
    paths = [master_tiles['path'][master_tiles['geometry'] == c].values[0] for c in intersects]
    #subfolders = [master_tiles['subfolder'][master_tiles['geometry'] == c].values[0] for c in intersects]
    
    #tilelist = [os.path.sep.join([paths[i], subfolders[i], f]) for i, f in enumerate(fnames)]
    tilelist = [os.path.sep.join([paths[i], f]) for i, f in enumerate(fnames)]
    # create a temporary VRT from the tiles
    gdal.BuildVRT(os.path.sep.join([indir, 'tmp_{}.vrt'.format(dname)]), tilelist, resampleAlg='bilinear')
    # print(os.path.sep.join([os.path.abspath(indir), 'tmp_{}.vrt'.format(dname)]))
    return os.path.sep.join([os.path.abspath(indir), 'tmp_{}.vrt'.format(dname)])


def main():
    np.seterr(all='ignore')
    parser = argparse.ArgumentParser(description="Run MMASTER post-processing bias corrections, given external elevation data.")
    # things to add: input directory, master dem and/or elevation data
    # optional: output directory, number of processors, land mask, glacier mask, number of jitter iterations?
    # tolerance for fit?
    parser.add_argument('master_tiles', type=str, help='Shapefile of master DEM footprints to be used for bias correction')
    parser.add_argument('indir', action='store', nargs='+', type=str,
                        help="directory/directories where final, georeferenced images are located.")
    parser.add_argument('-s', '--slavedem', type=str, default=None,
                        help='(optional) name of DEM to correct. By default, mmaster_bias_correction.py\
                        looks for MMASTER DEMs of the form AST_L1A_..._Z.tif')
    parser.add_argument('-a', '--exc_mask', type=str, default=None,
                        help='exclusion mask. Areas inside of this shapefile (i.e., glaciers)\
                        will not be used for coregistration [None]')
    parser.add_argument('-b', '--inc_mask', type=str, default=None,
                        help='inclusion mask. Areas outside of this mask (i.e., water)\
                             will not be used for coregistration. [None]')
    parser.add_argument('-n', '--nproc', type=int, default=1, help='number of processors to use [1].')
    parser.add_argument('-o', '--outdir', type=str, default='biasrem',
                        help='directory to output files to (creates if not already present). [.]')
    parser.add_argument('-p', '--points', action='store_true', default=False,
                        help="process assuming that master DEM is point elevations [False].")
    parser.add_argument('-l', '--log', action='store_true', default=False,
                        help="write output to a log file rather than printing to the screen [False].")
    args = parser.parse_args()
    
    # figure out if we have one image or many
    print('Number of image directories given: {}'.format(len(args.indir)))
    
    # open master_tiles, set up a search tree
    master_tiles = gpd.read_file(args.master_tiles)
    s = STRtree([f for f in master_tiles['geometry'].values])
    
    # only go through the trouble of setting up multiprocessing
    # if we have more than one directory to work on.
    if args.nproc > 1 and len(args.indir) > 1:
        if args.nproc > mp.cpu_count():
            print("{} cores specified to use, but I could only find \
                  {} cores on this machine, so I'll use those.'".format(args.nproc, mp.cpu_count()))
            args.nproc = mp.cpu_count
        pool = mp.Pool(args.nproc, maxtasksperchild=1)
        # get a dictionary of arguments for each of the different DEMs,
        # starting with the common arguments (master dem, glacier mask, etc.)
        arg_dict = {'glacmask': args.exc_mask, 
                    'landmask': args.inc_mask, 
                    'pts': args.points, 
                    'out_dir': args.outdir,
                    'return_geoimg': False,
                    'write_log': True}
        u_args = [{'work_dir': d,
                   'slv_dem': '{}_Z.tif'.format(d)} for i, d in enumerate(args.indir)]
        for d in u_args:
            d.update(arg_dict)
        
        pool_args = [(u, master_tiles, s) for u in u_args]
        pool.map(batch_wrapper, pool_args, chunksize=1)
        pool.close()
        pool.join()
    else:
        odir = os.getcwd()
        for i, indir in enumerate(args.indir):
            mst_dem = get_tiles(indir, master_tiles, s)
            print('Running bias correction on {}'.format(indir))
            os.chdir(indir)
            #print(os.getcwd())
            if args.slavedem is None:
                flist = glob('AST*_Z.tif')
                this_slavedem = flist[0]
                mmaster_bias_removal(mst_dem,
                                     this_slavedem,
                                     glacmask=args.exc_mask,
                                     landmask=args.inc_mask,
                                     out_dir=args.outdir,
                                     pts=args.points,
                                     return_geoimg=False,
                                     write_log=args.log)
            else:
                mmaster_bias_removal(mst_dem,
                                     args.slavedem,
                                     glaciermask=args.exc_mask,
                                     landmask=args.inc_mask,
                                     out_dir=args.outdir,
                                     pts=args.points,
                                     return_geoimg=False,
                                     write_log=args.log)
            os.chdir(odir)
    
    # clean up after ourselves, remove the vrts we created.
    for d in args.indir:
        os.remove(os.path.sep.join([d, 'tmp_{}.vrt'.format(d)]))

    
if __name__ == "__main__":
    main()