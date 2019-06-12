#! /usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
# import geopandas as gpd
import numpy as np
import datetime as dt
import gdal
import netCDF4
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from osgeo import osr
from pybob.GeoImg import GeoImg
from mmaster_tools import reproject_geometry
# from pybob.coreg_tools import dem_coregistration


# some helper function definitions
def read_stats(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    stats = lines[0].strip('[ ]\n').split(', ')
    after = [float(s) for s in lines[-1].strip('[ ]\n').split(', ')]
    
    return dict(zip(stats, after))


def parse_date(fname):
     bname = os.path.splitext(os.path.basename(fname))[0]
     splitname = bname.split('_')
     if splitname[0] == 'AST':
         datestr = splitname[2][3:]
         datefmt = '%m%d%Y%H%M%S'
     elif splitname[0] in ['SETSM', 'SDMI', 'SPOT', 'Map', 'IDEM', 'S0']:
         datestr = splitname[2]
         datefmt = '%Y%m%d'
     else:
         print("I don't recognize how to parse date information from {}.".format(fname))
         return None
     return dt.datetime.strptime(datestr, datefmt)


def get_common_bbox(args):
    fprints = []
    tmp = GeoImg(args.filelist[0])
    this_epsg = tmp.epsg
    fp = Polygon(tmp.find_corners(mode='xy'))
    for f in args.filelist[1:]:
        tmp = GeoImg(f)
        fp = Polygon(tmp.find_corners(mode='xy'))
        fprints.append(reproject_geometry(fp, tmp.epsg, this_epsg))
    common_fp = cascaded_union(fprints)

    if tmp.epsg != args.epsg:
        common_fp = reproject_geometry(common_fp, tmp.epsg, args.epsg)
    
    bbox = common_fp.envelope
    x, y = bbox.boundary.coords.xy
    return min(x), max(x), min(y), max(y)


def create_crs_variable(epsg, nco):
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(epsg)
    
    sref_wkt = sref.ExportToWkt()

    crso = nco.createVariable('crs', 'S1')
    crso.long_name = sref_wkt.split(',')[0].split('[')[-1].replace('"', '').replace(' / ', ' ')
    
    if 'PROJCS' in sref_wkt:
        split1 = sref_wkt.split(',PROJECTION')
        split2 = split1[1].split('],')
        split3 = split1[0].split(',GEOGCS[')[1]

        crso.grid_mapping_name = split2[0].strip('["').lower()

        params = [s.split('[')[-1] for s in split2 if 'PARAMETER' in s]
        for p in params:
            if p.split(',')[0].strip('"') == 'scale_factor':
                exec('crso.{} = {}'.format('proj_scale_factor', float(p.split(',')[1])))
            else:
                exec('crso.{} = {}'.format(p.split(',')[0].strip('"'), float(p.split(',')[1])))
        ustr = [s.split('[')[1].split(',')[0].strip('"') for s in split2 if 'UNIT' in s][0]
        crso.units = ustr
        crso.spheroid = split3.split(',SPHEROID[')[0].split(',')[0].strip('"').replace(' ','')
        crso.semi_major_axis = float(split3.split(',SPHEROID')[1].split(',')[1])
        crso.inverse_flattening = float(split3.split(',SPHEROID')[1].split(',')[2])
        crso.datum = split3.split(',SPHEROID[')[0].split(',DATUM[')[-1].strip('"')
        crso.longitude_of_prime_meridian = int(split3.split(',PRIMEM')[-1].split(',')[1])
        if crso.grid_mapping_name == 'polar_stereographic':
            crso.scale_factor_at_projection_origin = crso.proj_scale_factor
            crso.standard_parallel = crso.latitude_of_origin
        #crso.spheroid = 'WGS84'
        #crso.datum = 'WGS84'
        crso.spatial_ref = sref_wkt
    else:   # have to figure out what to do with a non-projected system...
        pass

    return crso


def create_nc(first_img, args):
    nrows, ncols = first_img.img.shape

    nco = netCDF4.Dataset(args.outfile, 'w', clobber=args.clobber)
    nco.createDimension('x', ncols)
    nco.createDimension('y', nrows)
    nco.createDimension('time', None)
    nco.Conventions='CF-1.7'
    nco.description = "Stack of co-registered DEMs produced using MMASTER (+ other sources). \n" +\
                      "MMASTER scripts and documentation: https://github.com/luc-girod/MMASTER-workflows \n" +\
                      "pybob source and documentation: https://github.com/iamdonovan/pybob"
    nco.history = "Created " + time.ctime(time.time())
    nco.source = "Robert McNabb (robertmcnabb@gmail.com)" 
    
    to = nco.createVariable('time', 'f4', ('time'))
    to.units = 'days since 1900-01-01'
    to.calendar = 'standard'
    to.standard_name = 'date'
    
    xo = nco.createVariable('x', 'f4', ('x'))
    xo.units = 'm'
    xo.standard_name = 'projection_x_coordinate'
    xo.axis = 'X'
    
    yo = nco.createVariable('y', 'f4', ('y'))
    yo.units = 'm'
    yo.standard_name = 'projection_y_coordinate'
    yo.axis = 'Y'
    
    return nco, to, xo, yo


def main():
    parser = argparse.ArgumentParser(description="Create a NetCDF stack of DEMs.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filelist', action='store', type=str, nargs='+',
                        help='List of DEM files to read and stack.')
    parser.add_argument('-extent', action='store', metavar=('xmin', 'xmax', 'ymin', 'ymax'),
                        type=float, nargs=4, help='Extent of output DEMs to write.')
    parser.add_argument('-res', action='store', type=float, default=None,
                        help='pixel resolution (in meters)')
    parser.add_argument('-epsg', action='store', type=int, default=None,
                        help='Target EPSG code. Default is taken from first (chronological) DEM.')
    parser.add_argument('-o', '--outfile', action='store', type=str, default='mmaster_stack.nc',
                        help='Output NetCDF file to create [mmaster_stack.nc]')
    parser.add_argument('-c', '--clobber', action='store_true',
                        help='Overwrite any existing file [False]')
    parser.add_argument('-u', '--uncert', action='store_true',
                        help='Read stable terrain statistics for each DEM from [filename].txt')
    args = parser.parse_args()
    
    # have to generate ncols, nrows, etc. from the first filename
    
    datelist = [parse_date(f) for f in args.filelist]
    sorted_inds = np.argsort(datelist)
    
    tmp_img = GeoImg(args.filelist[sorted_inds[0]])
    
    if args.res is None:
        args.res = np.round(tmp_img.dx) # make sure that we have a nice resolution for gdal
    
    if args.epsg is None:
        args.epsg = tmp_img.epsg
    
    if args.extent is not None:
        xmin, xmax, ymin, ymax = args.extent
    else:
        xmin, xmax, ymin, ymax = get_common_bbox(args)
    
    # now, reproject the first image to the extent, resolution, and coordinate system needed.
    dest = gdal.Warp('', tmp_img.gd, format='MEM', dstSRS='EPSG:{}'.format(args.epsg), 
                     xRes=args.res, yRes=args.res, outputBounds=(xmin, ymin, xmax, ymax),
                     resampleAlg=gdal.GRA_Bilinear)    
    
    first_img = GeoImg(dest)
    first_img.filename = args.filelist[sorted_inds[0]]
    # NetCDF assumes that coordinates are the cell center
    if first_img.is_area():
        first_img.to_point()
    #first_img.info()
    
    nco, to, xo, yo = create_nc(first_img, args)
    create_crs_variable(first_img.epsg, nco)
    # crso.GeoTransform = ' '.join([str(i) for i in first_img.gd.GetGeoTransform()])
    
    #maxchar = max([len(f.rsplit('.tif', 1)[0]) for f in args.filelist])
    go = nco.createVariable('dem_names', str, ('time',))
    go.long_name = 'Source DEM Filename'
    
    zo = nco.createVariable('z', 'f4', ('time', 'y', 'x'), fill_value=-9999)
    zo.units = 'meters'
    zo.long_name = 'Height above WGS84 ellipsoid'
    zo.grid_mapping = 'crs'
    zo.coordinates = 'x y'
    zo.set_auto_mask(True)
    
    if args.uncert:
        uo = nco.createVariable('uncert', 'f4', ('time',))
        uo.long_name = 'RMSE of stable terrain differences.'
        uo.units = 'meters'
        
    x, y = first_img.xy(grid=False)
    xo[:] = x
    yo[:] = y
    to[0] = datelist[sorted_inds[0]].toordinal() - dt.date(1900, 1, 1).toordinal()
    go[0] = args.filelist[sorted_inds[0]].rsplit('.tif', 1)[0]
    zo[0, :, :] = first_img.img
    if args.uncert:
        stats = read_stats(args.filelist[sorted_inds[0]].replace('.tif', '.txt'))
        uo[0] = stats['RMSE']
    
    for i, ind in enumerate(sorted_inds[1:]):
        print(args.filelist[ind])
        img = GeoImg(args.filelist[ind])
        img = img.reproject(first_img)
        if img.is_area():  # netCDF assumes coordinates are the cell center
            img.to_point()       
        if args.uncert:
            stats = read_stats(args.filelist[ind].replace('.tif', '.txt'))
            uo[i+1] = stats['RMSE']
        to[i+1] = datelist[ind].toordinal() - dt.date(1900, 1, 1).toordinal()
        go[i+1] = args.filelist[ind].rsplit('.tif', 1)[0]
        zo[i+1, :, :] = img.img
    
    nco.close()

if __name__ == "__main__":
    main()