from __future__ import print_function
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import time
import errno
# import geopandas as gpd
import numpy as np
import datetime as dt
import gdal
import netCDF4
import geopandas as gpd
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from shapely.strtree import STRtree
from osgeo import osr
from pybob.GeoImg import GeoImg
from mmaster_tools import reproject_geometry
from pybob.coreg_tools import dem_coregistration


def mkdir_p(out_dir):
    try:
        os.makedirs(out_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise    


def read_stats(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    stats = lines[0].strip('[ ]\n').split(', ')
    after = [float(s) for s in lines[-1].strip('[ ]\n').split(', ')]
    
    return dict(zip(stats, after))


def parse_date(fname, datestr=None, datefmt=None):
     bname = os.path.splitext(os.path.basename(fname))[0]
     splitname = bname.split('_')
     if datestr is None and datefmt is None:
         if splitname[0] == 'AST':
             datestr = splitname[2][3:]
             datefmt = '%m%d%Y%H%M%S'
         elif splitname[0] in ['SETSM', 'SDMI', 'SPOT5', 'Map', 'IDEM', 'S0', 'USGS', 'CDEM', 'TDM1']:
             datestr = splitname[2]
             datefmt = '%Y%m%d'
         elif splitname[0] in ['aerodem']:
             datestr = splitname[1]
             datefmt = '%Y%m%d'
         else:
             print("I don't recognize how to parse date information from {}.".format(fname))
             return None
     return dt.datetime.strptime(datestr, datefmt)


def get_footprints(filelist, epsg=None):
    fprints = []
    if epsg is not None:
        this_epsg = epsg
    else:
        tmp = GeoImg(filelist[0])
        this_epsg = tmp.epsg
    
    for f in filelist:
        tmp = GeoImg(f)
        fp = Polygon(tmp.find_corners(mode='xy'))
        fprints.append(reproject_geometry(fp, tmp.epsg, this_epsg))
    
    return fprints

        
def get_common_bbox(filelist, epsg=None):
    fprints = get_footprints(filelist, epsg)
    
    common_fp = cascaded_union(fprints)
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


def create_nc(img, outfile='mmaster_stack.nc', clobber=False, t0=None):
    nrows, ncols = img.shape

    nco = netCDF4.Dataset(outfile, 'w', clobber=clobber)
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
    if t0 is None:
        to.units = 'days since 1900-01-01'
    else:
        to.units = 'days since {}'.format(np.datetime_as_string(t0))
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


def get_tiles(bounds, master_tiles, s, name):
    res = s.query(bounds)
    # get only the results that intersect our buffered footprint more than 10% of its area
    intersects = [c for c in res if bounds.intersection(c).area / c.area > 0.1]
    fnames = [master_tiles['filename'][master_tiles['geometry'] == c].values[0] for c in intersects]
    paths = [master_tiles['path'][master_tiles['geometry'] == c].values[0] for c in intersects]
    subfolders = [master_tiles['subfolder'][master_tiles['geometry'] == c].values[0] for c in intersects]
    
    tilelist = [os.path.sep.join([paths[i], subfolders[i], f]) for i, f in enumerate(fnames)]
    # create a temporary VRT from the tiles
    gdal.BuildVRT('{}_mst.vrt'.format(name), tilelist, resampleAlg='bilinear')
    # print(os.path.sep.join([os.path.abspath(indir), 'tmp_{}.vrt'.format(dname)]))
    return '{}_mst.vrt'.format(name)


def create_mmaster_stack(filelist, extent=None, res=None, epsg=None, outfile='mmaster_stack.nc', 
                         clobber=False, uncert=False, coreg=False, mst_tiles=None,
                         exc_mask=None, inc_mask=None, outdir='tmp', filt_dem=None):
    """

    """    
    if extent is not None:
        if type(extent) in [list, tuple]:
            xmin, xmax, ymin, ymax = extent
        elif type(extent) is Polygon:
            x, y = extent.boundary.coords.xy
            xmin, xmax = min(x), max(x)
            ymin, ymax = min(y), max(y)
        else:
            raise ValueError('extent should be a list, tuple, or shapely.Polygon')
    else:
        xmin, xmax, ymin, ymax = get_common_bbox(filelist, epsg)
    
    
    if coreg and mst_tiles is not None:
        master_tiles = gpd.read_file(mst_tiles)
        s = STRtree([f for f in master_tiles['geometry'].values])        
        bounds = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        mst_vrt = get_tiles(bounds, master_tiles, s, outdir)
        mst = GeoImg(mst_vrt)

    # check if each footprint falls within our given extent, and if not - remove from the list.
    #fprints = get_footprints(filelist, epsg)
    #red_filelist = [f for i, f in filelist if extPoly.intersects(fprints[i])]

    datelist = np.array([parse_date(f) for f in filelist])
    sorted_inds = np.argsort(datelist)

    print(filelist[sorted_inds[0]])
    tmp_img = GeoImg(filelist[sorted_inds[0]])

    if res is None:
        res = np.round(tmp_img.dx) # make sure that we have a nice resolution for gdal
    
    if epsg is None:
        epsg = tmp_img.epsg

    # now, reproject the first image to the extent, resolution, and coordinate system needed.
    dest = gdal.Warp('', tmp_img.gd, format='MEM', dstSRS='EPSG:{}'.format(epsg), 
                     xRes=res, yRes=res, outputBounds=(xmin, ymin, xmax, ymax),
                     resampleAlg=gdal.GRA_Bilinear)
    
    first_img = GeoImg(dest)
    first_img.filename = filelist[sorted_inds[0]]
    # NetCDF assumes that coordinates are the cell center
    if first_img.is_area():
        first_img.to_point()
    #first_img.info()
    
    nco, to, xo, yo = create_nc(first_img.img, outfile, clobber)
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
    
    if filt_dem is not None:
        filt_dem_img = GeoImg(filt_dem)
        filt_dem = filt_dem_img.reproject(first_img)
    
    if uncert:
        uo = nco.createVariable('uncert', 'f4', ('time',))
        uo.long_name = 'RMSE of stable terrain differences.'
        uo.units = 'meters'
        
    x, y = first_img.xy(grid=False)
    xo[:] = x
    yo[:] = y
    to[0] = datelist[sorted_inds[0]].toordinal() - dt.date(1900, 1, 1).toordinal()
    go[0] = os.path.basename(filelist[sorted_inds[0]]).rsplit('.tif', 1)[0]
    if coreg:
        NDV = tmp_img.NDV
        if tmp_img.is_area():
            tmp_img.to_point()

        _, img, _ = dem_coregistration(mst, tmp_img, glaciermask=exc_mask, landmask=inc_mask, outdir=outdir)
        dest = gdal.Warp('', img.gd, format='MEM', dstSRS='EPSG:{}'.format(epsg), 
                         xRes=res, yRes=res, outputBounds=(xmin, ymin, xmax, ymax),
                         resampleAlg=gdal.GRA_Bilinear, srcNodata=NDV, dstNodata=-9999)
        img = GeoImg(dest)
        if filt_dem is not None:
            valid = np.logical_and(img.img-filt_dem.img > -400,
                                   img.img-filt_dem.img < 1000)
            img.img[~valid] = np.nan
        zo[0, :, :] = img.img
        if uncert:
            stats = read_stats('{}/stats.txt'.format(outdir))
            uo[0] = stats['RMSE']
    else:
        zo[0, :, :] = first_img.img
        if uncert:
            stats = read_stats(filelist[sorted_inds[0]].replace('.tif', '.txt'))
            uo[0] = stats['RMSE']

    outind = 1
    for ind in sorted_inds[1:]:
        print(filelist[ind])
        img = GeoImg(filelist[ind])
        if img.is_area():  # netCDF assumes coordinates are the cell center
            img.to_point()
        if coreg:
            try:
                NDV = img.NDV
                _, img, _ = dem_coregistration(mst, img, glaciermask=exc_mask, landmask=inc_mask, outdir=outdir)
                dest = gdal.Warp('', img.gd, format='MEM', dstSRS='EPSG:{}'.format(epsg),
                                 xRes=res, yRes=res, outputBounds=(xmin, ymin, xmax, ymax),
                                 resampleAlg=gdal.GRA_Bilinear, srcNodata=NDV, dstNodata=-9999)
                img = GeoImg(dest)
                if filt_dem is not None:
                    valid = np.logical_and(img.img-filt_dem.img > -400,
                                           img.img-filt_dem.img < 1000)
                    img.img[~valid] = np.nan
                zo[outind, :, :] = img.img    
                if uncert:
                    stats = read_stats('{}/stats.txt'.format(outdir))
                    uo[outind] = stats['RMSE']
            except:
                continue
        
        else:
            img = img.reproject(first_img)
            zo[outind, :, :] = img.img
            if uncert:
                stats = read_stats(filelist[ind].replace('.tif', '.txt'))
                uo[outind] = stats['RMSE']
        to[outind] = datelist[ind].toordinal() - dt.date(1900, 1, 1).toordinal()
        go[outind] = os.path.basename(filelist[ind]).rsplit('.tif', 1)[0]
        zo[outind, :, :] = img.img
        outind += 1
    
    return nco
    