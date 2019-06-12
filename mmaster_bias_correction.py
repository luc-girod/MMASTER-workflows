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
from mmaster_tools import mmaster_bias_removal


def batch_wrapper(arg_dict):
    return mmaster_bias_removal(**arg_dict)

def main():
    np.seterr(all='ignore')
    parser = argparse.ArgumentParser(description="Run MMASTER post-processing bias corrections, given external elevation data.")
    # things to add: input directory, master dem and/or elevation data
    # optional: output directory, number of processors, land mask, glacier mask, number of jitter iterations?
    # tolerance for fit?
    parser.add_argument('masterdem', type=str, help='path to master DEM/elevations to be used for bias correction')
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
    parser.add_argument('-e', '--single_equation', action='store_false', default=True,
                        help="Switch to control the fitting process, either to solve biases in 1 step \
                             or to solve jitter in 2 steps [True].")
    args = parser.parse_args()
    
    # figure out if we have one image or many
    print('Hello')
    print('Number of image directories given: {}'.format(len(args.indir)))
    # only go through the trouble of setting up multiprocessing
    # if we have more than one directory to work on.
    if args.nproc > 1 and len(args.indir) > 1:
        if args.nproc > mp.cpu_count():
            print("{} cores specified to use, but I could only find \
                  {} cores on this machine, so I'll use those.'".format(args.nproc, mp.cpu_count()))
            args.nproc = mp.cpu_count
        pool = mp.Pool(args.nproc)
        # get a dictionary of arguments for each of the different DEMs,
        # starting with the common arguments (master dem, glacier mask, etc.)
        arg_dict = {'mst_dem': args.masterdem, 
                    'glacmask': args.exc_mask, 
                    'landmask': args.inc_mask, 
                    'pts': args.points, 
                    'out_dir': args.outdir,
                    'return_geoimg': False,
                    'write_log': True,
                    'robust': args.single_equation}
        u_args = [{'work_dir': d, 'slv_dem': '{}_Z.tif'.format(d)} for d in args.indir]
        for d in u_args:
            d.update(arg_dict)
        
        pool.map(batch_wrapper, u_args)
        pool.close()
        pool.join()
    else:
        odir = os.getcwd()
        for indir in args.indir:
            print('Running bias correction on {}'.format(indir))
            os.chdir(indir)
            #print(os.getcwd())
            if args.slavedem is None:
                flist = glob('AST*_Z.tif')
                this_slavedem = flist[0]
                mmaster_bias_removal(args.masterdem,
                                     this_slavedem,
                                     glacmask=args.exc_mask,
                                     landmask=args.inc_mask,
                                     out_dir=args.outdir,
                                     pts=args.points,
                                     return_geoimg=False,
                                     write_log=args.log,
                                     robust=args.single_equation)
            else:
                mmaster_bias_removal(args.masterdem,
                                     args.slavedem,
                                     glaciermask=args.exc_mask,
                                     landmask=args.inc_mask,
                                     out_dir=args.outdir,
                                     pts=args.points,
                                     return_geoimg=False,
                                     write_log=args.log,
                                     robust=args.single_equation)
            os.chdir(odir)
        

if __name__ == "__main__":
    main()