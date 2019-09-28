"""
pymmaster.mmaster_tools provides most of the routines used for removing jitter-related bias in ASTER DEMs produced using
`MicMac ASTER`_.

.. _MicMac ASTER: https://www.mdpi.com/2072-4292/9/7/704
"""
from __future__ import print_function
# from future_builtins import zip
from functools import partial
import os
import sys
from glob import glob
import errno
import pyproj
import numpy as np
import numpy.polynomial.polynomial as poly
import gdal
import ogr
import fiona
import fiona.crs
import matplotlib.pylab as plt
import pandas as pd
import scipy.optimize as optimize
import time
import zipfile
import shutil
from skimage.morphology import disk
from scipy.ndimage.morphology import binary_opening, binary_dilation
from scipy.ndimage.filters import median_filter
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry.polygon import Polygon, orient
from shapely.geometry import mapping, LineString, Point
from shapely.ops import cascaded_union, transform
from pybob.coreg_tools import dem_coregistration, false_hillshade, get_slope, create_stable_mask
from pybob.GeoImg import GeoImg
from pybob.image_tools import nanmedian_filter
from pybob.plot_tools import plot_shaded_dem
from pybob.bob_tools import mkdir_p
from mpl_toolkits.axes_grid1 import make_axes_locatable


def extract_file_from_zip(fn_zip_in, filename_in, fn_file_out):
    with zipfile.ZipFile(fn_zip_in) as zip_file:

        for member in zip_file.namelist():
            filename = os.path.basename(member)
            if filename != filename_in:
                # skip directories
                continue

            # copy file (taken from zipfile's extract)
            source = zip_file.open(member)
            target = open(fn_file_out, 'wb')
            with source, target:
                shutil.copyfileobj(source, target)

def create_zip_from_flist(list_fn,fn_zip):

    with zipfile.ZipFile(fn_zip,'w') as zip_file:
        for fn in list_fn:
            zip_file.write(fn,arcname=os.path.basename(fn),compress_type=zipfile.ZIP_DEFLATED)

        zip_file.close()

def clean_coreg_dir(out_dir, cdir):
    search_str = glob(os.path.sep.join([out_dir, cdir, '*.tif']))
    for f in search_str:
        os.remove(f)


def make_mask(inpoly, pts, raster_out=False):
    """
    Create a True/False mask to determine whether input points are within a polygon.

    :param inpoly: Input polygon to use to create mask.
    :param pts: Either a list of (x,y) coordinates, or a GeoImg. If a list of (x,y) coordinates,
        make_mask uses shapely.polygon.contains to determine if the point is within inpoly.
        If pts is a GeoImg, make_mask uses ogr and gdal to "rasterize" inpoly to the GeoImg raster.
    :param raster_out: Kind of output to return. If True, pts must be a GeoImg.
    :type inpoly: shapely Polygon
    :type pts: array-like, pybob.GeoImg
    :type raster_out: bool
    :returns mask: An array which is True where pts lie within inpoly and False elsewhere.
    """
    if not raster_out:
        mask = []
        for pt in pts:
            mask.append(inpoly.contains(Point(pt)))
        return mask
    else:
        if type(pts) is not GeoImg:
            raise TypeError('raster_out is requested, but pts is not a GeoImg')
        # use gdal, ogr
        # first, create a layer for the polygon to live in.
        drv = ogr.GetDriverByName('Memory')
        dst_ds = drv.CreateDataSource('out')
        masklayer = dst_ds.CreateLayer('test', geom_type=ogr.wkbMultiPolygon, srs=pts.spatialReference)
        feature = ogr.Feature(masklayer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkt(inpoly.wkt))
        #        feature.SetGeometry(ogr.CreateGeometryFromWkt(inpoly.crs_wkt))
        masklayer.CreateFeature(feature)
        feature.Destroy()
        # now, create the raster to burn the mask to.
        masktarget = gdal.GetDriverByName('MEM').Create('', pts.npix_x, pts.npix_y, 1, gdal.GDT_Byte)
        masktarget.SetGeoTransform((pts.xmin, pts.dx, 0, pts.ymax, 0, pts.dy))
        masktarget.SetProjection(pts.proj_wkt)
        masktarget.GetRasterBand(1).Fill(0)
        gdal.RasterizeLayer(masktarget, [1], masklayer)
        mask = masktarget.GetRasterBand(1).ReadAsArray()
        mask[mask != 0] = 1
        return mask == 1


def make_group_id(inmat, grpid):
    """
    Make a unique ID for statistical analysis.

    :param inmat: Input array to create ID for.
    :param grpid: Array of input group IDs
    :type inmat: array-like
    :type grpid: array-like

    :returns outmat: Output of group IDs
    """
    return np.floor(inmat / grpid) * grpid


def get_group_statistics(invar, indata, indist=500):
    """
    Calculate statistics of groups of pixels grouped by along-track distance.

    :param invar: Input array that determines how to group data (i.e., along-track distance)
    :param indata: Data to calculate group statistics for.
    :param indist: Distance by which to group pixels.
    :type invar: array-like
    :type indata: array-like
    :type indist: float

    :returns grp_stats: group statistics for input data.
    """
    xxid = make_group_id(invar, indist)  # across track coordinates (grouped 500m)
    # yyid = make_group_id(yyr,500) # along track coordinates (grouped 500m)

    # Calculate group statistics 
    mykeep = np.isfinite(indata)
    data = pd.DataFrame({'dH': indata[mykeep], 'XX': xxid[mykeep]})
    xxgrp = data['dH'].groupby(data['XX']).describe()

    return xxgrp


def reproject_geometry(src_data, src_crs, dst_crs):
    """
    Reproject a geometry object from one coordinate system to another.

    :param src_data: geometry object to reproject
    :param src_crs: proj4 description of source CRS
    :param dst_crs: proj4 description of destination CRS
    :type src_data: shapely geometry
    :type src_crs: str, dict
    :type dst_crs: str, dict

    :returns dst_data: reprojected data.
    """
    # unfortunately this requires pyproj>1.95, temporary fix to avoid shambling dependencies in mmaster_environment
    src_proj = pyproj.Proj(src_crs)
    dst_proj = pyproj.Proj(dst_crs)

    project = partial(pyproj.transform, src_proj, dst_proj)
    return transform(project, src_data)


def nmad(data):
    """
    Calculate the Normalized Median Absolute Deviation (NMAD) of the input dataset.

    :param data: input data on which to calculate NMAD
    :type data: array-like

    :returns nmad: NMAD of input dataset.
    """
    return np.nanmedian(np.abs(data) - np.nanmedian(data))


def get_aster_footprint(gran_name, proj4='+units=m +init=epsg:4326', indir=None, polyout=True):
    """
    Create shapefile of ASTER footprint from .met file

    :param gran_name: ASTER granule name to use; assumed to also be the folder in which .met file(s) are stored.
    :param proj4: proj4 representation for coordinate system to use [default: '+units=m +init=epsg:4326', WGS84 Lat/Lon].
    :param indir: Directory to search in [default: current working directory].
    :param polyout: Create a shapefile of the footprint in the input directory [True].

    :type gran_name: str
    :type proj4: str, dict
    :type indir: str
    :type polyout: bool
    :returns footprint: shapely Polygon representing ASTER footprint, reprojected to given CRS.
    """

    ### MADE AN ADJUSTMENT TO ASSUME THAT THE .MET FILE IS IN THE CURRENT FOLDER OF OPERATION
    if indir is None:
        metlist = glob(os.path.abspath('*.met'))
    else:
        metlist = glob(os.path.abspath(os.path.sep.join([indir, '*.met'])))

    if polyout:
        schema = {'properties': [('id', 'int')], 'geometry': 'Polygon'}
        outshape = fiona.open(gran_name + '_Footprint.shp', 'w', crs=proj4,
                              driver='ESRI Shapefile', schema=schema)

    footprints = []
    for m in metlist:
        clean = [line.strip() for line in open(m).read().split('\n')]

        if os.path.sep in m:
            m = m.split(os.path.sep)[-1]

        latinds = [i for i, line in enumerate(clean) if 'GRingPointLatitude' in line]
        loninds = [i for i, line in enumerate(clean) if 'GRingPointLongitude' in line]

        latlines = clean[latinds[0]:latinds[1] + 1]
        lonlines = clean[loninds[0]:loninds[1] + 1]

        lonvalstr = lonlines[2]
        latvalstr = latlines[2]

        lats = [float(val) for val in latvalstr.strip('VALUE  = ()').split(',')]
        lons = [float(val) for val in lonvalstr.strip('VALUE  = ()').split(',')]

        coords = list(zip(lons, lats))
        footprints.append(Polygon(coords))

    footprint = cascaded_union(footprints)
    footprint = footprint.simplify(0.0001)
    outprint = reproject_geometry(footprint, {'init': 'epsg:4326'}, proj4)
    if polyout:
        outshape.write({'properties': {'id': 1}, 'geometry': mapping(outprint)})
        outshape.close()
    return outprint


def orient_footprint(fprint):
    """
    Orient ASTER footprint coordinates to be clockwise, with upper left coordinate first.

    :param fprint: footprint to orient
    :type fprint: shapely polygon
    :returns o_fprint: re-oriented copy of input footprint.
    """
    # orient the footprint coordinates so that they are clockwise
    fprint = orient(fprint, sign=-1)
    x, y = fprint.boundary.coords.xy
    x = x[:-1]  # drop the last coordinate, which is a duplicate of the first
    y = y[:-1]
    # as long as the footprints are coming from the .met file, the upper left corner 
    # will be the maximum y value.
    upper_left = np.argmax(y)
    new_inds = list(range(upper_left, len(x))) + list(range(0, upper_left))
    return Polygon(list(zip(np.array(x)[new_inds], np.array(y)[new_inds])))


def get_track_angle(fprint, track_dist):
    """
    Calculate the angle made by the ASTER flight track from scene footprint.

    :param fprint: Footprint of ASTER scene
    :param track_dist: Distance along flight track within scene at which to calculate angle.
    :type fprint: shapely Polygon
    :type track_dist: float

    :returns track_angle: angle, in degrees, of flight track
    """
    # orient the footprint
    fprint = orient_footprint(fprint)
    x, y = fprint.boundary.coords.xy
    # upper_left = np.argmax(y)
    upper_right = np.argmax(x)
    lower_right = np.argmin(y)
    lower_left = np.argmin(x)

    lside = range(len(x) - 1, lower_left - 1, -1)
    rside = range(upper_right, lower_right + 1)

    left_side = LineString(list(zip(np.array(x)[lside], np.array(y)[lside])))
    right_side = LineString(list(zip(np.array(x)[rside], np.array(y)[rside])))

    lproj = left_side.interpolate(track_dist)
    rproj = right_side.interpolate(track_dist)
    # get the angle of the line formed by connecting lproj, rproj
    dx = lproj.x - rproj.x
    dy = lproj.y - rproj.y

    return 90 + np.rad2deg(np.arctan(dx / dy))


def preprocess(mst_dem, slv_dem, glacmask=None, landmask=None, work_dir='.', out_dir='biasrem', pts=False):
    """
    Pre-process ASTER scene to enable cross- and along-track corrections. Co-registers the
    ASTER (slave) and external (master) DEMs/ICESat, and shifts the orthoimage and correlation mask
    based on the offset calculated. Results are saved by default in a folder called 'coreg' which
    is moved to the 'bias_removal' folder at the end of this function

    :param mst_dem: Path to filename or GeoImg dataset representing external "master" DEM or ICESat dataset.
    :param slv_dem: Path to filename or GeoImg dataset representing ASTER DEM
    :param glacmask: Path to shapefile representing areas to exclude from co-registration
        consideration (i.e., glaciers).
    :param landmask: Path to shapefile representing areas to include in co-registration
        consideration (i.e., stable ground/land).
    :param work_dir: Working directory to use [Assumes '.'].
    :param out_dir: Output directory for the new coregistration folder
    :param pts: True if mst_dem is point data, False if mst_dem is a DEM.
    :type mst_dem: str, pybob.Geoimg, pybob.ICESat
    :type slv_dem: str, pybob.GeoImg
    :type glacmask: str
    :type landmask: str
    :type cwd: str
    :type out_dir: str

    :returns mst_coreg, slv_coreg, shift_params: co-registered master, slave datasets, as well as a tuple containing
        x,y,z shifts calculated during co-registration process.
    """
    # if the output directory does not exist, create it.
    # out_dir = os.path.sep.join([work_dir, out_dir])
    # print(out_dir)
    try:
        os.makedirs(out_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise
    # Get the ASTER names
    ast_name = slv_dem.rsplit('_Z.tif')[0]
    mask_name = os.path.sep.join([work_dir, '{}_CORR.tif'.format(ast_name)])
    slv_name = os.path.sep.join([work_dir, '{}_filtZ.tif'.format(ast_name)])
    # filter DEM using correlation mask !!! NEED to update to allow for either GeoIMG or pathname as INPUT
    mask_raster_threshold(slv_dem, mask_name, slv_name, 60, np.float32)

    # co-register master, slave
    coreg_dir = os.path.sep.join([out_dir, 'coreg'])
    mst_coreg, slv_coreg, shift_params = dem_coregistration(mst_dem, slv_name, glaciermask=glacmask,
                                                            landmask=landmask, outdir=coreg_dir, pts=pts)[0:3]
    if shift_params == -1:
        return mst_coreg, slv_coreg, shift_params
    #    print(slv_coreg.filename)
    # remove coreg folder, save slv as *Zadj1.tif, but save output.pdf
    #    shutil.move(os.path.sep.join([out_dir, 'CoRegistration_Results.pdf']),
    #                os.path.sep.join([cwd, 'CoRegistration_Results.pdf']))
    #    shutil.move(os.path.sep.join([out_dir, 'coreg_params.txt']), os.path.sep.join([cwd, 'coreg_params.txt']))
    #    shutil.rmtree(out_dir)

    ### EXPORT and write files. Move directory to main Bias Corrections directory
    ### *** NOTE: GeoImg.write replaces the mask !!!***
    slv_coreg.write(os.path.sep.join([out_dir, '{}_Z_adj.tif'.format(ast_name)]))
    # shift ortho, corr masks, save as *adj1.tif
    ortho = GeoImg(ast_name + '_V123.tif')
    corr = GeoImg(ast_name + '_CORR.tif')

    ortho.shift(shift_params[0], shift_params[1])
    ortho.write(os.path.sep.join([out_dir, '{}_V123_adj.tif'.format(ast_name)]), dtype=np.uint8)

    corr.shift(shift_params[0], shift_params[1])
    corr.write(os.path.sep.join([out_dir, '{}_CORR_adj.tif'.format(ast_name)]), dtype=np.uint8)

    plt.close("all")

    return mst_coreg, slv_coreg, shift_params


def mask_raster_threshold(rasname, maskname, outfilename, threshold=50, datatype=np.float32):
    """
    Filter values in an input raster if the correlation mask raster falls below a threshold value. Removes noise from
    low-correlation areas (i.e., clouds) using a binary opening operation.

    :param rasname: path to raster (i.e., DEM) to mask, or GeoImg object.
    :param maskname: path to correlation raster to determine mask values, or GeoImg object.
    :param outfilename: filename to write masked raster to.
    :param threshold: correlation value to determine masked values.
    :param datatype: datatype to save raster as.

    :type rasname: str, pybob.GeoImg
    :type maskname: str, pybob.GeoImg
    :type outfilename: str
    :type threshold: float
    :type datatype: numpy datatype

    :returns masked_ras: GeoImg of masked input raster.
    """
    myras = GeoImg(rasname)
    mymask = GeoImg(maskname)
    mymask.img[mymask.img < threshold] = 0

    rem_open = binary_opening(mymask.img, structure=disk(5))
    myras.img[~rem_open] = np.nan
    myras.write(outfilename, driver='GTiff', dtype=datatype)

    return myras


# can probably just import this from pybob.coreg_tools, no?
def rmse(indata):
    """
    Return root mean square of input differences.

    :param indata: differences to calculate RMSE of
    :type indata: array-like

    :returns data_rmse: RMSE of input data.
    """
    # add 3x std dev filter
    #    indata[np.abs(indata)>3*np.nanstd(indata)] = np.nan

    myrmse = np.sqrt(np.nanmean(np.square(np.asarray(indata))))
    return myrmse


def calculate_dH(mst_dem, slv_dem, pts):
    """
    Calculate difference between master and slave DEM. Master and slave DEMs must have the same shape. First,
    differences are calculated, then data are filtered using a 5x5 median filter, and differences greater than 100 m,
    or differences on high (>50deg) or low (< 0.5deg) slopes are removed.

    :param mst_dem: master DEM (i.e., external DEM/ICESat data)
    :param slv_dem: slave DEM (i.e., ASTER scene)
    :param pts: True if master DEM is ICESat data, False if master DEM is a DEM.

    :returns: dH: filtered elevation differences between master and slave DEMs.
    """
    if not pts:
        zupdate = np.ma.array(mst_dem.img.data - slv_dem.img.data, mask=slv_dem.img.mask)
        #        zupdate2 = np.ma.array(ndimage.median_filter(zupdate, 7), mask=slv_dem.img.mask)

        # JIT_FILTER_FUNCTION DOESNT WORK IN WINDOWS
        # Create conditional for windows system. jit_filter_function is currently 
        # failing due to Windows incompatibility with numba
        # https://github.com/numba/numba/issues/2578
        #
        if os.name == 'nt':
            # first get the mask of nans
            kernel = np.ones((3, 3), np.uint8)
            mymask = np.multiply(np.isnan(zupdate), 1, dtype='uint8')
            mymask2 = binary_dilation(mymask, kernel)
            zupdate2 = median_filter(zupdate, 7)
            zupdate2[mymask2 == 1] = np.nan
            zupdate2 = np.ma.array(zupdate2, mask=slv_dem.img.mask)

        else:  # on UNIX
            zupdate2 = np.ma.array(nanmedian_filter(zupdate, size=7), mask=slv_dem.img.mask)

        dH = slv_dem.copy(new_raster=zupdate2)

        master_mask = isinstance(mst_dem.img, np.ma.masked_array)
        slave_mask = isinstance(slv_dem.img, np.ma.masked_array)

        myslope = get_slope(slv_dem)
        fmask = np.logical_or.reduce((np.greater(np.abs(dH.img), 100),
                                      np.less(myslope.img, 0.5),
                                      np.greater(myslope.img, 25)))
        # if we have two masked arrays, take the logical 'or' of those masks
        # and apply it to dH
        if master_mask and slave_mask:
            smask = np.logical_or(mst_dem.img.mask, slv_dem.img.mask)
        elif master_mask:
            smask = mst_dem.img.mask
        elif slave_mask:
            smask = slv_dem.img.mask
        else:  # needs to have an option where none of the above
            smask = np.zeros(dH.img.shape, dtype=bool)  # should be all false
        mask = np.logical_or(smask, fmask)
        dH.mask(mask)
    else:
        # NEED TO CHECK THE MASKING
        slave_pts = slv_dem.raster_points2(mst_dem.xy, nsize=3, mode='cubic')
        dH = mst_dem.elev - slave_pts

        myslope = get_slope(slv_dem)
        slope_pts = myslope.raster_points2(mst_dem.xy, nsize=3, mode='cubic')

        fmask = np.logical_or.reduce((np.greater(np.abs(dH), 100), np.less(slope_pts, 0.5), np.greater(slope_pts, 25)))

        dH[fmask] = np.nan
    return dH


def get_xy_rot(dem, myang):
    """
    Rotate x, y axes of image to get along- and cross-track distances.

    :param dem: DEM to get x,y positions from.
    :param myang: angle by which to rotate axes
    :type dem: pybob GeoImg
    :type myang: float

    :returns xxr, yyr: arrays corresponding to along (x) and cross (y) track distances.
    """
    # creates matrices for along and across track distances from a reference dem and a raster angle map (in radians)

    xx, yy = dem.xy(grid=True)
    xx = xx - np.min(xx)
    yy = yy - np.min(yy)
    # xxr = np.multiply(xx,np.rad2deg(np.cos(myang))) + np.multiply(-1*yy,np.rad2deg(np.sin(myang)))
    # yyr = np.multiply(xx,np.rad2deg(np.sin(myang))) + np.multiply(yy,np.rad2deg(np.cos(myang)))
    xxr = np.multiply(xx, np.cos(myang)) + np.multiply(-1 * yy, np.sin(myang))
    yyr = np.multiply(xx, np.sin(myang)) + np.multiply(yy, np.cos(myang))

    # TO USE FOR INITIALIZING START AT ZERO
    xxr = xxr - np.nanmin(xxr)
    yyr = yyr - np.nanmin(yyr)

    plt.figure(figsize=(5, 5))
    plt.imshow(xxr, interpolation='nearest')
    # plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(yyr, interpolation='nearest')
    # plt.show()

    return xxr, yyr


def get_atrack_coord(mst_dem, myangN, myangB):
    """
    Createx numpy arrays for along- and cross-track distances from DEM and angle maps.

    :param dem: DEM to get x,y positions from
    :param myangN: MMASTER 3N track angle (i.e., track angle in nadir image)
    :param myangB: MMASTER 3B track angle (i.e., track angle in back-looking image)
    :type dem: pybob GeoImg
    :type myangN: pybob GeoImg
    :type myangB: pybob GeoImg

    :returns yyn, yyb: along-track distances in 3N and 3B track directions.
    """

    myangN = np.deg2rad(myangN.img)
    myangB = np.deg2rad(myangB.img)

    xx, yy = mst_dem.xy(grid=True)
    xx = xx - np.min(xx)
    yy = yy - np.min(yy)
    # xxr = np.multiply(xx,np.rad2deg(np.cos(myang))) + np.multiply(-1*yy,np.rad2deg(np.sin(myang)))
    # yyr = np.multiply(xx,np.rad2deg(np.sin(myang))) + np.multiply(yy,np.rad2deg(np.cos(myang)))
    # xxr = np.multiply(xx,np.cos(myang)) + np.multiply(-1*yy,np.sin(myang))

    yyn = np.multiply(xx, np.sin(myangN)) + np.multiply(yy, np.cos(myangN))
    yyb = np.multiply(xx, np.sin(myangB)) + np.multiply(yy, np.cos(myangB))

    yyn = yyn - np.nanmin(yyn)
    yyb = yyb - np.nanmin(yyb)
    plt.figure(figsize=(5, 5))
    plt.imshow(yyn, interpolation='nearest')
    # plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(yyb, interpolation='nearest')
    # plt.show()

    return yyn, yyb


def get_fit_variables(mst_dem, slv_dem, xxn, pts, xxb=None):
    """
    Get input variables for bias fitting.

    :param mst_dem: Master DEM to use for fitting
    :param slv_dem: Slave DEM to use for fitting
    :param xxn: along- or cross-track distance in the nadir track direction
    :param pts: True if mst_dem is ICESat points, False if mst_dem is a GeoImg.
    :param xxb: along- or cross-track direction in the back-looking track direction. If not provided, fit variables are
        provided only for one angle.
    :type mst_dem: pybob GeoImg, ICESat
    :type slv_dem: pybob GeoImg
    :type xxn: array-like
    :type pts: bool
    :type xxb: array-like
    :returns xx, dH, grp_xx, grp_dH, xx2, grp_sts: track distance, elevation differences, group distances, group
        median dH group values, group distances in the back-looking direction, group statistics.
    """
    if not pts:
        dHmat = calculate_dH(mst_dem, slv_dem, pts)
        dH = dHmat.img.reshape((1, dHmat.img.size))
        xx = xxn.reshape((1, xxn.size))
        if xxb is not None:
            xx2 = xxb.reshape((1, xxb.size))
        masked = isinstance(dH, np.ma.masked_array)
        if masked:
            xx = np.ma.masked_array(xxn.reshape((1, xxn.size)), dH.mask)
            if xxb is not None:
                xx2 = np.ma.masked_array(xx2, dH.mask)
                xx2 = xx2.compressed()
            xx = xx.compressed()
            dH = dH.compressed()
    elif pts:
        XXR = slv_dem.copy(new_raster=xxn)
        xx = XXR.raster_points2(mst_dem.xy, nsize=3, mode='cubic')
        dH = calculate_dH(mst_dem, slv_dem, pts)
        if xxb is not None:
            XXR2 = slv_dem.copy(new_raster=xxb)
            xx2 = XXR2.raster_points2(mst_dem.xy, mode='cubic')
    # Mask and filter (remove outliers)
    mynan = np.logical_or.reduce((np.invert(np.isfinite(dH)),
                                  np.invert(np.isfinite(xx)),
                                  (np.abs(dH) > np.nanstd(dH) * 3)))

    dH = np.squeeze(dH[~mynan])
    xx = np.squeeze(xx[~mynan])

    # get group statistics of dH, and create matrix with same shape as orig_data
    grp_sts = get_group_statistics(xx, dH, indist=500)
    grp_xx = grp_sts.index.values
    grp_dH = grp_sts['50%'].values

    if xxb is None:
        xx2 = np.nan
    else:
        xx2 = np.squeeze(xx2[~mynan])

    return xx, dH, grp_xx, grp_dH, xx2, grp_sts


def fitfun_polynomial(xx, params):
    #    myval=0
    #    for i in np.arange(0,params.size):
    #        myval = myval + params[i]*(xx**i)
    # myval=myval + params[i]*(xx**i)
    return sum([p * (np.divide(xx, 1000) ** i) for i, p in enumerate(params)])


def robust_polynomial_fit(xx, yy):
    """
    Given sample data xx, yy, compute a robust polynomial fit to the data. Order is chosen automatically by comparing
    residuals for multiple fit orders.

    :param xx: input x data (typically across-track distance)
    :param yy: input y data (typically elevation differences)
    :type xx: array-like
    :type yy: array-like

    :returns coefs, order: polynomial coefficients and order for the best-fit polynomial
    """
    print("Original Sample size :", yy.size)
    # mykeep=np.isfinite(yy) and np.isfinite(xx)
    #    mykeep = np.logical_and(np.isfinite(yy), np.isfinite(xx))
    mykeep = np.logical_and.reduce((np.isfinite(yy), np.isfinite(xx), (np.abs(yy) < np.nanstd(yy) * 3)))
    xx = xx[mykeep]
    yy = yy[mykeep]

    print("Final Sample size :", yy.size)
    print("Remaining NaNs :", np.sum(np.isnan(yy)))
    sampsize = min(int(0.15 * xx.size), 25000)  # np.int(np.floor(xx.size*0.25))
    if xx.size > sampsize:
        mysamp = np.random.randint(0, xx.size, sampsize)
    else:
        mysamp = np.arange(0, xx.size)

    plt.figure(figsize=(7, 5), dpi=200)
    # fig.suptitle(title, fontsize = 14)
    plt.plot(xx[mysamp], yy[mysamp], '^', ms=1, color='0.5', rasterized=True, fillstyle='full')

    myorder = 6
    mycost = np.empty(myorder)
    coeffs = np.zeros((myorder, myorder + 1))
    xnew = np.arange(np.nanmin(xx), np.nanmax(xx), 1000)

    def errfun(p, xx, yy):
        return fitfun_polynomial(xx, p) - yy

    for deg in np.arange(1, myorder + 1):
        #        p0 = np.zeros(deg + 1)
        p0 = poly.polyfit(np.divide(xx[mysamp], 1000), yy[mysamp], deg)
        print("Initial Parameters: ", p0)
        #        lbb = np.zeros(deg + 1)-1000
        #        ubb = np.zeros(deg + 1)+1000

        #        myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]))
        myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', loss='soft_l1',
                                           f_scale=0.5, ftol=1E-8, xtol=1E-8)
        #        print("Status: ", myresults.status)
        print("Polynomial degree - ", deg, " --> Status: ", myresults.success, " - ", myresults.status)
        print(myresults.message)
        print("Lowest cost:", myresults.cost)
        print("Parameters:", myresults.x)

        mycost[deg - 1] = myresults.cost
        coeffs[deg - 1, 0:myresults.x.size] = myresults.x

        mypred = fitfun_polynomial(xnew, myresults.x)
        plt.plot(xnew, mypred)

    fidx = mycost.argmin()
    plt.ylim(-75, 75)

    # This is to check whether percent improvement is a better way to choose the best fit. 
    # For now, comment out... 
    #    perimp=np.divide(mycost[:-1]-mycost[1:],mycost[:-1])*100
    #    fmin=np.asarray(np.where(perimp>5))
    #    if fmin.size!=0:
    #        fidx = fmin[0,-1]+1
    # print('fidx: {}'.format(fidx))

    print("Polynomial Order Selected: ", fidx + 1)

    return np.trim_zeros(coeffs[fidx], 'b'), fidx + 1


def polynomial_fit(x, y):
    """
    [DEPRECATED] A polynomial search function for orders 1-6 given dependent (x) and independent (y)
    variables. Uses the numpy polynomial package. 
    """
    # edge removal
    #    x = x[5:-5]
    #    y = y[5:-5]

    max_poly_order = 6

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, '.')
    rmse = np.empty(max_poly_order - 1)
    coeffs = np.zeros((max_poly_order - 1, max_poly_order + 1))
    xnew = np.arange(np.nanmin(x), np.nanmax(x), 1000)
    for deg in np.arange(2, max_poly_order + 1):
        # print(deg)
        p, r = poly.polyfit(x, y, deg, full=True)
        p2 = poly.polyval(xnew, p)
        plt.plot(xnew, p2)
        coeffs[deg - 2, 0:p.size] = p
        rmse[deg - 2] = np.sqrt(np.divide(r[0], y.size - deg))

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(2, max_poly_order + 1), rmse)

    # Choose order of polynomial - 2 options 
    # a) lowest RMSE or 
    # b) by checking the percent improvemnt (more robust?)

    # a) lowest RMSE
    fidx = rmse.argmin()

    # b) [DEFAULT??] find the highest order which gave a 5% improvemnt in the RMSE
    perimp = np.divide(rmse[:-1] - rmse[1:], rmse[:-1]) * 100
    # print(perimp)
    # print(rmse)
    fmin = np.asarray(np.where(perimp > 5))
    if fmin.size != 0:
        fidx = fmin[0, -1] + 1
    # print('fidx: {}'.format(fidx))

    print("Polynomial Order Selected: ", fidx + 2)
    return coeffs[fidx], rmse[fidx]


def fitfun_sumofsin(xx, p):
    """
    The Base function for the sum of sines fitting to elevation differences in the along track (xx) direction
    of the ASTER flightpath. DEPRECATED - uses only one angle instead of two. see fitfun_sumofsin_2angle
    """
    #    myval = 0
    #    for bb in np.arange(0, p.size - 1, 3):
    #        myval = myval + p[bb] * np.sin(p[bb + 1] * xx + p[bb + 2])
    #
    p = np.asarray(p)
    aix = np.arange(0, p.size, 3)
    bix = np.arange(1, p.size, 3)
    cix = np.arange(2, p.size, 3)
    #    if len(xx.shape) == 1:
    #        myval = np.sum(p[aix] * np.sin(p[bix] * xx[:,np.newaxis] + p[cix]),axis=1)
    #    elif len(xx.shape) == 2:
    #        myval = np.sum(p[aix] * np.sin(p[bix] * xx[:,:,np.newaxis] + p[cix]),axis=2)
    if len(xx.shape) == 1:
        myval = np.sum(p[aix] * np.sin(np.divide(2 * np.pi, p[bix]) * np.divide(xx[:, np.newaxis], 1000) + p[cix]),
                       axis=1)
    elif len(xx.shape) == 2:
        myval = np.sum(p[aix] * np.sin(np.divide(2 * np.pi, p[bix]) * np.divide(xx[:, :, np.newaxis], 1000) + p[cix]),
                       axis=2)

    return myval


def fitfun_sumofsin_2angle(xxn, xxb, p):
    """
    Fit sum of sines function for two track angles.

    :param xxn: along-track distance in nadir direction
    :param xxb: along-track distance in back-looking direction
    :param p: sum of sines parameters
    :type xxn: array-like
    :type yyn: array-like
    :type p: array-like

    :returns sum_of_sines: sum of sines evaluated at the given along-track distances.
    """
    p = np.squeeze(np.asarray(p))
    aix = np.arange(0, p.size, 6)
    bix = np.arange(1, p.size, 6)
    cix = np.arange(2, p.size, 6)
    dix = np.arange(3, p.size, 6)
    eix = np.arange(4, p.size, 6)
    fix = np.arange(5, p.size, 6)

    if len(xxn.shape) == 1:
        myval = np.sum(p[aix] * np.sin(np.divide(2 * np.pi, p[bix]) *
                                       np.divide(xxn[:, np.newaxis], 1000) +
                                       p[cix]) + p[dix] * np.sin(np.divide(2 * np.pi, p[eix]) *
                                                                 np.divide(xxb[:, np.newaxis], 1000) + p[fix]), axis=1)
    elif len(xxn.shape) == 2:
        myval = np.sum(p[aix] * np.sin(np.divide(2 * np.pi, p[bix]) *
                                       np.divide(xxn[:, :, np.newaxis], 1000) +
                                       p[cix]) + p[dix] * np.sin(np.divide(2 * np.pi, p[eix]) *
                                                                 np.divide(xxb[:, :, np.newaxis], 1000) + p[fix]),
                       axis=2)
    return myval


def huber_loss(z):
    out = np.asarray(np.square(z) * 1.000)
    out[np.where(z > 1)] = 2 * np.sqrt(z[np.where(z > 1)]) - 1
    return out.sum()


def soft_loss(z):  # z is residual
    return 2 * (np.sqrt(1 + z) - 1)  # SOFT-L1 loss function (reduce the weight of outliers)


def costfun_sumofsin(p, xxn, yy, xxb=None, myscale=0.5):
    if xxb is not None:
        myval = fitfun_sumofsin_2angle(xxn, xxb, p)
    else:
        myval = fitfun_sumofsin(xxn, p)
    # DEFINE THE COST FUNCTION
    #    myerr = RMSE(yy - myval)
    #    myerr = nmad(yy - myval)
    #    myerr = np.sum(np.abs(yy-myval))    # called MAE or L1 loss function
    #    myerr = np.linalg.norm(yy-myval)
    #    myerr = np.sqrt(np.sum((yy-myval) ** 2))
    #    myerr = huber_loss(yy-myval)    # HUBER loss function (reduce the weight of outliers)
    #    myerr = np.sum((np.sqrt(1+np.square(yy-myval))-1))    # SOFT-L1 loss function (reduce the weight of outliers)
    # myscale = 0.5
    myerr = np.sum(np.square(myscale) * soft_loss(np.square(np.divide(yy - myval, myscale))))
    #    myerr = np.sum( np.square(myscale)*2*(np.sqrt(1+np.square(np.divide(yy-myval,myscale)))-1) ) 
    # SOFT-L1 loss function  with SCALING
    return myerr


def plot_bias(xx, dH, grp_xx, grp_dH, title, pp, pmod=None, smod=None, plotmin=None, txt=None):
    """
    data : original data as numpy array (:,2), x = 1col,y = 2col
    gdata : grouped data as numpy array (:,2)
    pmod,smod are two model options to plot, numpy array (:,1)
    """

    mykeep = np.isfinite(dH)
    xx = xx[mykeep]
    dH = dH[mykeep]
    sampsize = min(int(0.15 * xx.size), 25000)
    if xx.size > sampsize:
        mysamp = np.random.randint(0, xx.size, sampsize)
    else:
        mysamp = np.arange(0, xx.size)
    # mysamp = mysamp.astype(np.int64) #huh?

    # title = 'Cross'
    fig = plt.figure(figsize=(7, 5), dpi=200)
    # fig.suptitle(title + 'track bias', fontsize=14)
    plt.title(title + 'track bias', fontsize=14)
    if plotmin is None:
        plt.plot(xx[mysamp], dH[mysamp], '^', ms=0.75, color='0.5', rasterized=True, fillstyle='full',
                 label="Raw [samples]")
        plt.plot(grp_xx, grp_dH, '-', ms=2, color='0.15', label="Grouped Median")
    else:
        plt.plot(grp_xx, grp_dH, '^', ms=1, color='0.5', rasterized=True, fillstyle='full', label="Grouped Median")

    # xx2 = np.linspace(np.min(grp_xx), np.max(grp_xx), 1000)
    if pmod is not None:
        plt.plot(pmod[0], pmod[1], 'r-', ms=2, label="Basic Fit")
    if smod is not None:
        plt.plot(smod[0], smod[1], 'm-', ms=2, label="SumOfSines-Fit")

    plt.plot(grp_xx, np.zeros(grp_xx.size), 'k-', ms=1)

    plt.xlim(np.min(grp_xx), np.max(grp_xx))
    # plt.ylim(-200,200)
    ymin, ymax = plt.ylim((np.nanmean(dH[mysamp])) - 2 * np.nanstd(dH[mysamp]),
                          (np.nanmean(dH[mysamp])) + 2 * np.nanstd(dH[mysamp]))

    # plt.axis([0, 360, -200, 200])
    plt.xlabel(title + ' track distance [meters]')
    plt.ylabel('dH [meters]')
    plt.legend(loc='upper right')
    #    plt.legend(('Raw [samples]', 'Grouped Median', 'Fit'), loc=1)

    if txt is not None:
        plt.text(0.05, 0.05, txt, fontsize=12, fontweight='bold', color='black',
                 family='monospace', transform=plt.gca().transAxes)

    # plt.show()
    pp.savefig(fig, dpi=300)


def final_histogram(dH0, dH1, dH2, dHfinal, pp):
    fig = plt.figure(figsize=(7, 5), dpi=600)
    plt.title('Elevation difference histograms', fontsize=14)
    if isinstance(dH0, np.ma.masked_array):
        dH0 = dH0.compressed()
        dH1 = dH1.compressed()
        dH2 = dH2.compressed()
        dHfinal = dHfinal.compressed()
    dH0 = np.squeeze(np.asarray(dH0[np.logical_and.reduce((np.isfinite(dH0), (np.abs(dH0) < np.nanstd(dH0) * 3)))]))
    dH1 = np.squeeze(np.asarray(dH1[np.logical_and.reduce((np.isfinite(dH1), (np.abs(dH1) < np.nanstd(dH1) * 3)))]))
    dH2 = np.squeeze(np.asarray(dH2[np.logical_and.reduce((np.isfinite(dH2), (np.abs(dH2) < np.nanstd(dH2) * 3)))]))
    dHfinal = np.squeeze(np.asarray(dHfinal[np.logical_and.reduce((np.isfinite(dHfinal),
                                                                   (np.abs(dHfinal) < np.nanstd(dHfinal) * 3)))]))

    if dH0[np.isfinite(dH0)].size < 2000:
        mybins = 40
    else:
        mybins = 100

    j1, j2 = np.histogram(dH0[np.isfinite(dH0)], bins=mybins, range=(-60, 60))
    jj1, jj2 = np.histogram(dH1[np.isfinite(dH1)], bins=mybins, range=(-60, 60))
    jjj1, jjj2 = np.histogram(dH2[np.isfinite(dH2)], bins=mybins, range=(-60, 60))
    k1, k2 = np.histogram(dHfinal[np.isfinite(dHfinal)], bins=mybins, range=(-60, 60))

    stats0 = [np.nanmean(dH0), np.nanmedian(dH0), np.nanstd(dH0), rmse(dH0)]
    stats_fin = [np.nanmean(dHfinal), np.nanmedian(dHfinal), np.nanstd(dHfinal), rmse(dHfinal)]

    plt.plot(j2[1:], j1, 'k-', linewidth=2, label="original")
    plt.plot(jj2[1:], jj1, 'b-', linewidth=2, label="After X-track")
    plt.plot(jjj2[1:], jjj1, 'm-', linewidth=2, label="After A-track (low freq)")
    plt.plot(k2[1:], k1, 'r-', linewidth=2, label="After A-track (all freq)")

    plt.xlabel('Elevation difference [meters]')
    plt.ylabel('Number of samples')
    plt.xlim(-50, 50)

    # numwidth = max([len('{:.1f} m'.format(xadj)), len('{:.1f} m'.format(yadj)), len('{:.1f} m'.format(zadj))])
    plt.text(0.05, 0.90, 'Mean: ' + ('{:.1f} m'.format(stats0[0])),
             fontsize=8, fontweight='bold', color='black', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, 'Median: ' + ('{:.1f} m'.format(stats0[1])),
             fontsize=8, fontweight='bold', color='black', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.80, 'Std dev.: ' + ('{:.1f} m'.format(stats0[2])),
             fontsize=8, fontweight='bold', color='black', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.75, 'RMSE: ' + ('{:.1f} m'.format(stats0[3])),
             fontsize=8, fontweight='bold', color='black', family='monospace', transform=plt.gca().transAxes)

    plt.text(0.05, 0.60, 'Mean: ' + ('{:.1f} m'.format(stats_fin[0])),
             fontsize=8, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.55, 'Median: ' + ('{:.1f} m'.format(stats_fin[1])),
             fontsize=8, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.50, 'Std dev.: ' + ('{:.1f} m'.format(stats_fin[2])),
             fontsize=8, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)
    plt.text(0.05, 0.45, 'RMSE: ' + ('{:.1f} m'.format(stats_fin[3])),
             fontsize=8, fontweight='bold', color='red', family='monospace', transform=plt.gca().transAxes)

    plt.legend(loc='upper right')

    pp.savefig(fig, bbox_inches='tight', dpi=200)


def correct_cross_track_bias(mst_dem, slv_dem, inang, pp, pts=False):
    # calculate along/across track coordinates
    myang = np.deg2rad(inang.img)
    xxr, yyr = get_xy_rot(slv_dem, myang)  # across,along track coordinates calculated from angle map

    # arrange the dependent (dH) and independent variables (ANGLE) into vectors
    # ALSO, filters the dH > threshold (40), and provides grouped statistics... 
    xx, dH, grp_xx, grp_dH, _, _ = get_fit_variables(mst_dem, slv_dem, xxr, pts)

    # #
    # Need conditional to check for large enough sample size... ?
    # #

    # POLYNOMIAL FITTING - here using my defined robust polynomial fitting
    #    pcoef, myorder = robust_polynomial_fit(grp_xx, grp_dH)
    xx2 = np.linspace(np.min(grp_xx), np.max(grp_xx), 1000)
    pcoef, myorder = robust_polynomial_fit(xx, dH)
    polymod = fitfun_polynomial(xx, pcoef)
    polymod_grp = fitfun_polynomial(xx2, pcoef)
    polyres = rmse(dH - polymod)
    print("Cross track robust Polynomial RMSE (all data): ", polyres)

    #   # USING POLYFIT With ALL/GROUPED data
    #    pcoef, _ = polynomial_fit(grp_xx,grp_dH) #mean
    #    pcoef2, _ = polynomial_fit(xx,dH) # USE ALL DATA
    #    polymod=poly.polyval(xx,pcoef2)
    #    polyres=RMSE(dH-polymod)
    #    print("Cross track standard Polynomial RMSE (all data): ", polyres)

    mytext = "Polynomial order: " + np.str(myorder)
    plot_bias(xx, dH, grp_xx, grp_dH, 'Cross', pp, pmod=(xx2, polymod_grp), txt=mytext)

    # Generate correction for DEM
    #    out_corr = poly.polyval(xxr, pcoef)
    out_corr = fitfun_polynomial(xxr, pcoef)

    # Correct DEM
    zupdate = np.ma.array(slv_dem.img + out_corr, mask=slv_dem.img.mask)  # shift in z
    slv_dem = slv_dem.copy(new_raster=zupdate)

    return slv_dem, out_corr, pcoef


def get_filtered_along_track(mst_dem, slv_dem, ang_maps, pts):
    ang_mapN, ang_mapB = ang_maps
    # across,along track coordinates calculated from angle map
    xxn_mat, xxb_mat = get_atrack_coord(slv_dem, ang_mapN, ang_mapB)

    # arrange the dependent (dH) and independent variables (ANGLE) into vectors
    # ALSO, filters the dH > threshold (40), and provides grouped statistics...     
    xxn, dH, grp_xx, grp_dH, xxb, grp_stats = get_fit_variables(mst_dem, slv_dem, xxn_mat, pts, xxb=xxb_mat)

    # # # # # # # # # # #
    # Need conditional to check for enough sample size... HERE it is.. ->
    xxid = make_group_id(xxn, 500)
    # percent sample size for the groups
    psize = np.divide(grp_stats['count'].values, np.sum(grp_stats['count'].values)) * 100
    pthresh = 0.1
    # create mask for dh and xx values    
    myix = np.isin(xxid, grp_xx[psize < pthresh], invert=True).flatten()
    # mask group values
    grp_dH = np.delete(grp_dH, [psize < pthresh])
    grp_xx = np.delete(grp_xx, [psize < pthresh])
    # # # # # # # # # # #

    yy = dH
    print("Original Sample Size: ", np.where(np.isfinite(xxn))[0].size)
    mykeep = np.logical_and.reduce((np.isfinite(yy),
                                    np.isfinite(xxn),
                                    np.isfinite(xxb),
                                    (np.abs(yy) < np.nanstd(yy) * 2.5),
                                    myix))
    xxn = np.squeeze(xxn[mykeep])
    xxb = np.squeeze(xxb[mykeep])
    yy = np.squeeze(yy[mykeep])
    print("Filtered Sample Size: ", xxn.size)

    return grp_xx, grp_dH, xxn, xxb, yy, xxn_mat, xxb_mat


def get_sample(vsize, n=15000):
    # sampsize = np.int(np.floor(xx.size*0.25)) # for use as a percentage
    sampsize = min(int(0.15 * vsize), n)
    if vsize > sampsize:
        mysamp = np.random.randint(0, vsize, sampsize)
    else:
        mysamp = np.arange(0, vsize)
    return mysamp


def correct_along_track_bias(mst_dem, slv_dem, ang_mapN, ang_mapB, pp, pts, robust=True):
    # calculate along/across track coordinates
    # myang = np.deg2rad(np.multiply(inang,np.multiply(dH,0)+1))# generate synthetic angle image for testing
    grp_xx, grp_dH, xxn, xxb, yy, xxn_mat, xxb_mat = get_filtered_along_track(mst_dem, slv_dem, (ang_mapN, ang_mapB),
                                                                              pts)
    # fig = plt.figure(figsize=(7, 5), dpi=200)
    # fig.suptitle(title, fontsize = 14)
    # plt.plot(xxn[mysamp], yy[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')

    # Define the bounds of the three sine wave coefficients to solve
    order = 2
    lb1 = [0, 50, -np.pi]  # long-wave amplitude, frequency, phase
    ub1 = [20, 140, np.pi]  #
    lb2 = [0, 20, -np.pi]  # mid-range
    ub2 = [10, 40, np.pi]
    lb3 = [0, 2, 0]  # jitter
    ub3 = [3, 10, 2 * np.pi]

    # RUN THE FIT ALL TOGETHER IF GOOD SAMPLING AND DATA AVAILABLE. IF SPATIAL SAMPLING IS LIMITED, THEN
    # IT IS BEST TO RUN THE ROBUST OPTION WHICH SOLVES FIRST THE LOW FREQUENCY AND THEN THE JITTER...
    if not robust:

        lbb = np.concatenate((np.tile(lb1, 2 * order), np.tile(lb2, 2 * order), np.tile(lb3, 2 * order)))
        ubb = np.concatenate((np.tile(ub1, 2 * order), np.tile(ub2, 2 * order), np.tile(ub3, 2 * order)))
        p0 = np.divide(lbb + ubb, 2)  # INITAL ESTIMATE
        # print(p0.size)
        # use the grouped statistics to get an initial estimate for the sum of sines fit
        # NOTE: only using one angle, needs two angles to be correct
        print("Fitting smoothed data to find initial parameters.")
        tt0 = time.time()
        init_args = dict(args=(grp_xx, grp_dH), method="L-BFGS-B",
                         bounds=optimize.Bounds(lbb, ubb), options={"ftol": 1E-4, "xtol": 1E-4})
        init_results = optimize.basinhopping(costfun_sumofsin, p0, disp=True,
                                             T=70,
                                             minimizer_kwargs=init_args)
        init_results = init_results.lowest_optimization_result
        # print(init_results.x)
        # use the initial estimate to start, USING two angles to get the lowest 2
        # frequencies for the sum of sines fit

        lbb = np.concatenate((np.tile(lb1, 2 * order), np.tile(lb2, 2 * order), np.tile(lb3, 2 * order)))
        ubb = np.concatenate((np.tile(ub1, 2 * order), np.tile(ub2, 2 * order), np.tile(ub3, 2 * order)))

        #        j2=np.reshape(init_results.x,(3,int(init_results.x.size/3)))
        #        j3=np.hstack((j2,j2))
        #        j4 = j3.reshape((j3.size,))
        #        p1 = j4 # Duplicate initial estimate for both angles
        #        print(lbb.size)
        #        print(p1.size)
        p1 = init_results.x

        mysamp = get_sample(xxn.size, n=20000)

        tt1 = time.time()
        print("Initial paramaters found in : ", (tt1 - tt0), " seconds")
        print("Sum of Sines Fitting using ", mysamp.size, "samples")
        minimizer_kwargs = dict(args=(xxn[mysamp], yy[mysamp], xxb[mysamp]),
                                method="L-BFGS-B",
                                bounds=optimize.Bounds(lbb, ubb),
                                options={"ftol": 1E-3, "xtol": 1E-2})
        myresults = optimize.basinhopping(costfun_sumofsin, p1, disp=True,
                                          T=700,  # niter_success=40,
                                          minimizer_kwargs=minimizer_kwargs)
        myresults = myresults.lowest_optimization_result
        tt2 = time.time()
        print("Sum of Sinses fitting finished in : ", (tt2 - tt1), " seconds")

        xxn2 = np.linspace(np.min(xxn), np.max(xxn), 1000)
        xxb2 = np.linspace(np.min(xxb), np.max(xxb), 1000)
        # mypred0 = fitfun_sumofsin(xxn2, init_results.x)
        mypred = fitfun_sumofsin_2angle(xxn2, xxb2, myresults.x)
        sinmod = fitfun_sumofsin_2angle(xxn_mat, xxb_mat, myresults.x)

        ## GET ONLY LOWER FREQUENCY RESULTS [REMNANT FROM ORIGINAL APPROACH TO SOLVE ALL FREQUENCIES AT ONCE]
        acoeff = myresults.x[:24]
        mypred2 = fitfun_sumofsin_2angle(xxn2, xxb2, acoeff)
        sinmod2 = fitfun_sumofsin_2angle(xxn_mat, xxb_mat, acoeff)

        plot_bias(xxn, yy, grp_xx, grp_dH, 'Along', pp, pmod=(xxn2, mypred2), smod=(xxn2, mypred))

        out_corr = np.reshape(sinmod, slv_dem.img.shape)
        out_corr2 = np.reshape(sinmod2, slv_dem.img.shape)
        jitter_corr = out_corr - out_corr2  # have to extract only the jitter component, even though we solved it all in one step.

        zupdate = np.ma.array(slv_dem.img + out_corr2, mask=slv_dem.img.mask)
        slv_dem_low = slv_dem.copy(new_raster=zupdate)

        zupdate2 = np.ma.array(slv_dem_low.img + jitter_corr, mask=slv_dem_low.img.mask)  # low frequencies
        slv_dem2 = slv_dem.copy(new_raster=zupdate2)

        fig2 = plt.figure(figsize=(7, 5), dpi=200)
        ax = fig2.gca()
        # fig.suptitle(title, fontsize = 14)
        plt1 = plt.imshow(out_corr)
        plt1.set_clim(np.nanmin(out_corr), np.nanmax(out_corr))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plt1, cax=cax)
        ax.set_title('Along-track Correction', fontsize=14)
        ax.set_xlabel('column', fontsize=14)
        ax.set_xlabel('row', fontsize=14)
        plt.tight_layout()
        pp.savefig(fig2, dpi=200)

        return (slv_dem_low, out_corr2, acoeff), (slv_dem2, jitter_corr, myresults.x)

    else:
        # get a random sample according to sample size
        mysamp = get_sample(xxn.size, n=20000)

        # Concatenate the bounds variables, This will determine the number of parameters
        # in the sum of sines equation, through the length of the p0, the initial estimate
        # lbb = np.concatenate((np.tile(lb1, 2 * order), np.tile(lb2, 2 * order), np.tile(lb3, 2 * order)))
        lbb = np.concatenate((np.tile(lb1, 2 * order), np.tile(lb2, 2 * order)))
        # ubb = np.concatenate((np.tile(ub1, 2 * order), np.tile(ub2, 2 * order), np.tile(ub3, 2 * order)))
        ubb = np.concatenate((np.tile(ub1, 2 * order), np.tile(ub2, 2 * order)))
        p0 = np.divide(lbb + ubb, 2)

        # use the grouped statistics to get an initial estimate for the sum of sines fit
        # NOTE: only using one angle, needs two angles to be correct
        print("Fitting smoothed data to find initial parameters.")
        tt0 = time.time()
        init_args = dict(args=(grp_xx, grp_dH), method="L-BFGS-B",
                         bounds=optimize.Bounds(lbb, ubb), options={"ftol": 1E-4})
        init_results = optimize.basinhopping(costfun_sumofsin, p0, disp=True,
                                             T=200,
                                             minimizer_kwargs=init_args)
        init_results = init_results.lowest_optimization_result
        # init_results = optimize.least_squares(costfun_sumofsin, p0, args=(grp_xx, grp_dH),
        #                                      method='dogbox', bounds=([lbb, ubb]), loss='linear',
        #                                      f_scale=5, ftol=1E-8, xtol=1E-8)
        # init_results = optimize.least_squares(costfun_sumofsin, p0, args=(grp_xx, grp_dH),
        #                                            method='trf', bounds=([lbb, ubb]), loss='soft_l1',
        #                                            f_scale=0.8, ftol=1E-6, xtol=1E-6)

        #    myresults0 = optimize.minimize(fitfun_sumofsin_2angle2, p0, args=(xxn[mysamp], xxb[mysamp], yy[mysamp]),
        #                                  bounds=optimize.Bounds(lbb,ubb), method='L-BFGS-B',
        #                                  options={'maxiter': 1000,'maxfun':1000, 'ftol':1E-8})

        # use the initial estimate to start, USING two angles to get the lowest 2
        # frequencies for the sum of sines fit
        tt1 = time.time()
        print("Initial paramaters found in : ", (tt1 - tt0), " seconds")
        print("Sum of Sines Fitting using ", mysamp.size, "samples")
        minimizer_kwargs = dict(args=(xxn[mysamp], yy[mysamp], xxb[mysamp]),
                                method="L-BFGS-B",
                                bounds=optimize.Bounds(lbb, ubb),
                                options={"ftol": 1E-4})
        myresults = optimize.basinhopping(costfun_sumofsin, init_results.x, disp=True,
                                          T=1000, niter_success=40,
                                          minimizer_kwargs=minimizer_kwargs)
        myresults = myresults.lowest_optimization_result
        tt2 = time.time()
        print("Sum of Sinses fitting finished in : ", (tt2 - tt1), " seconds")

        xxn2 = np.linspace(np.min(xxn), np.max(xxn), 1000)
        xxb2 = np.linspace(np.min(xxb), np.max(xxb), 1000)
        # mypred0 = fitfun_sumofsin(xxn2, init_results.x)
        mypred = fitfun_sumofsin_2angle(xxn2, xxb2, myresults.x)
        # init_fig = plt.figure(figsize=(7, 5), dpi=200)
        # plt.plot(xxn[mysamp], yy[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full', label='initial data')
        # plt.plot(grp_xx, grp_dH, 'k', label='grouped medians')
        # plt.plot(xxn2, mypred0, '-', ms=2, color='k', label='initial')
        # plt.plot(xxn2, mypred, '-', ms=2, color='r', label='final')
        # plt.legend()
        # pp.savefig(init_fig, dpi=200)

        ### GET ONLY LOWER FREQUENCY RESULTS [REMNANT FROM ORIGINAL APPROACH TO SOLVE ALL FREQUENCIES AT ONCE]
        # acoeff = myresults.x[:-18]
        # mypred2 = fitfun_sumofsin_2angle(xxn2, xxb2, acoeff)
        # sinmod2 = fitfun_sumofsin_2angle(xxn_mat, xxb_mat, acoeff)

        # plot_bias(orig_data,grp_data,mytype,pp)
        # plot_bias(xxn, yy, grp_xx, grp_dH, 'Along', pp, pmod=(xxn2, mypred2), smod=(xxn2, mypred))
        plot_bias(xxn, yy, grp_xx, grp_dH, 'Along', pp, smod=(xxn2, mypred))

        # out_corr2 = np.reshape(sinmod2, slv_dem.img.shape)
        # apply the low-frequency correction
        sinmod = fitfun_sumofsin_2angle(xxn_mat, xxb_mat, myresults.x)
        out_corr = np.reshape(sinmod, slv_dem.img.shape)

        zupdate = np.ma.array(slv_dem.img + out_corr, mask=slv_dem.img.mask)
        slv_dem_low = slv_dem.copy(new_raster=zupdate)

        fig2 = plt.figure(figsize=(7, 5), dpi=200)
        ax = fig2.gca()
        # fig.suptitle(title, fontsize = 14)
        plt1 = plt.imshow(out_corr)
        plt1.set_clim(np.nanmin(out_corr), np.nanmax(out_corr))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plt1, cax=cax)
        ax.set_title('Along-track Correction', fontsize=14)
        ax.set_xlabel('column', fontsize=14)
        ax.set_xlabel('row', fontsize=14)
        plt.tight_layout()
        pp.savefig(fig2, dpi=200)

        # now, estimate the jitter correction using the corrected slave dem
        grp_xx, grp_dH, xxn, xxb, yy, xxn_mat, xxb_mat = get_filtered_along_track(mst_dem, slv_dem_low,
                                                                                  (ang_mapN, ang_mapB), pts)
        sampsize = min(int(0.15 * xxn.size), 50000)
        if xxn.size > sampsize:
            mysamp = np.random.randint(0, xxn.size, sampsize)
        else:
            mysamp = np.arange(0, xxn.size)

        lb2 = [0, 20, 0]  # mid-range
        ub2 = [5, 40, 2 * np.pi]
        lb3 = [0, 3, 0]  # jitter
        ub3 = [3, 10, 2 * np.pi]
        lbb = np.concatenate((np.tile(lb2, 2 * order), np.tile(lb3, 2 * order)))
        ubb = np.concatenate((np.tile(ub2, 2 * order), np.tile(ub3, 2 * order)))

        p0 = np.divide(lbb + ubb, 2)

        if xxn.size < 10000:
            Tparam = 50
            myscale = 0.5
        else:
            Tparam = 500
            myscale = 0.1

        tt0 = time.time()
        minimizer_kwargs = dict(args=(xxn[mysamp], yy[mysamp], xxb[mysamp], myscale),
                                method="L-BFGS-B",
                                bounds=optimize.Bounds(lbb, ubb),
                                options={"ftol": 1E-4})
        jitter_res = optimize.basinhopping(costfun_sumofsin, p0, disp=True,
                                           T=Tparam, minimizer_kwargs=minimizer_kwargs)
        jitter_res = jitter_res.lowest_optimization_result
        tt1 = time.time()
        print("Sum of sines finished in : ", (tt1 - tt0), " seconds")

        #    xxn2 = np.linspace(np.min(xxn), np.max(xxn), 1000)
        jitt_pred = fitfun_sumofsin_2angle(xxn2, xxb2, jitter_res.x)
        plot_bias(xxn, yy, grp_xx, grp_dH, 'Jitter', pp, smod=(xxn2, jitt_pred))

        jitter_mod = fitfun_sumofsin_2angle(xxn_mat, xxb_mat, jitter_res.x)
        jitter_corr = np.reshape(jitter_mod, slv_dem_low.img.shape)
        # export slave dem with three constraing along track frequences
        zupdate2 = np.ma.array(slv_dem_low.img + jitter_corr, mask=slv_dem_low.img.mask)  # low frequencies
        slv_dem2 = slv_dem.copy(new_raster=zupdate2)

        fig3 = plt.figure(figsize=(7, 5), dpi=200)
        ax = fig3.gca()
        # fig.suptitle(title, fontsize = 14)
        plt1 = plt.imshow(jitter_corr)
        plt1.set_clim(np.nanmin(jitter_corr), np.nanmax(jitter_corr))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plt1, cax=cax)
        ax.set_title('Jitter-track Correction', fontsize=14)
        ax.set_xlabel('column', fontsize=14)
        ax.set_xlabel('row', fontsize=14)
        plt.tight_layout()
        pp.savefig(fig3, dpi=200)

        return (slv_dem_low, out_corr, myresults.x), (slv_dem2, jitter_corr, jitter_res.x)


def mmaster_bias_removal(mst_dem, slv_dem, glacmask=None, landmask=None,
                         pts=False, work_dir='.', tmp_dir=None, out_dir=None,
                         return_geoimg=True, write_log=False, zipped=False, robust=True):
    """
    Removes cross track and along track biases from MMASTER DEMs.

    :param mst_dem: Path to filename or GeoImg dataset representing "master" DEM or ICESat.
    :param slv_dem: Path to filename or GeoImg dataset representing "slave" DEM (developed for ASTER).
    :param glacmask: Path to shapefile representing points to exclude from co-registration
        consideration (i.e., glaciers).
    :param landmask: Path to shapefile representing points to include in co-registration
        consideration (i.e., stable ground/land).
    :param pts: If True, program assumes that masterDEM represents point data (i.e., ICESat),
        as opposed to raster data. Slope/aspect are then calculated from slaveDEM.
        masterDEM should be a string representing an HDF5 file continaing ICESat data.
    :param work_dir: Location where output files and directory should be saved [.]
    :param tmp_dir: Location where files are processed
    :param out_dir: Location to save bias removal outputs.
    :param return_geoimg: Return GeoImg objects of the corrected slave DEM and the co-registered master DEM [True]
    :param write_log: Re-direct stdout, stderr to a log file in the work directory [False]
    :param zipped: extract from zip archive, keep a minimum of output files (logs and pdfs only) [False]
    :param robust: solve low-frequency and jitter components of along-track bias separately. If spatial sampling is limited,
        this is a better approach than solving along-track as one single step.

    :type mst_dem: str, pybob.GeoImg, pybob.ICESat
    :type slv_dem: str, pybob.GeoImg
    :type glacmask: str
    :type landmask: str
    :type pts: bool
    :type work_dir: str
    :type out_dir: str
    :type return_geoimg: bool
    :type write_log: bool
    :type zipped: bool
    :type robust: bool

    :returns slv_corr, mst_coreg: corrected MMASTER DEM, co-registered master DEM (if return_geoimg)
    """
    orig_dir = os.getcwd()
    if tmp_dir is not None:
        if os.path.exists(os.path.join(tmp_dir,work_dir)):
            shutil.rmtree(os.path.join(tmp_dir,work_dir))
        shutil.copytree(os.path.join(orig_dir,work_dir),os.path.join(tmp_dir,work_dir))
        proc_dir=tmp_dir
    else:
        proc_dir=orig_dir

    final_dir = None
    if out_dir is None:
        out_dir = os.path.join(orig_dir,work_dir,'biasrem')
    else:
        if tmp_dir is None:
            out_dir = os.path.join(out_dir,work_dir)
        else:
            # we want the outputs to be written in the tmp directory, then moved to out_dir at the end, for speed
            final_dir = os.path.join(out_dir,work_dir)
            out_dir = os.path.join(proc_dir,work_dir,'tmp_biasrem')

    os.chdir(proc_dir)
    os.chdir(work_dir)

    if zipped:
        # we assume that the .zip has the same name as the L1A strip directory:
        strip_ref = '_'.join(os.path.basename(slv_dem).split('_')[:-1])
        # zip file
        fn_zip = os.path.join(proc_dir, work_dir, strip_ref + '.zip')
        # filenames to extract
        fn_slv = strip_ref + '_Z.tif'
        fn_corr = strip_ref + '_CORR.tif'
        fn_along3B = 'TrackAngleMap_3B.tif'
        fn_along3N = 'TrackAngleMap_3N.tif'
        fn_v123 = strip_ref + '_V123.tif'
        # files to extract to
        fn_slv_tif = os.path.join(proc_dir,work_dir, fn_slv)
        fn_corr_tif = os.path.join(proc_dir,work_dir, fn_corr)
        fn_along3B_tif = os.path.join(proc_dir,work_dir, fn_along3B)
        fn_along3N_tif = os.path.join(proc_dir,work_dir, fn_along3N)
        fn_v123_tif = os.path.join(proc_dir,work_dir, fn_v123)
        # extract
        extract_file_from_zip(fn_zip, fn_slv, fn_slv_tif)
        extract_file_from_zip(fn_zip, fn_corr, fn_corr_tif)
        extract_file_from_zip(fn_zip, fn_along3B, fn_along3B_tif)
        extract_file_from_zip(fn_zip, fn_along3N, fn_along3N_tif)
        extract_file_from_zip(fn_zip, fn_v123, fn_v123_tif)

        fn_filtz = os.path.join(proc_dir, work_dir, os.path.splitext(slv_dem)[0][:-2] + '_filtZ.tif')
        fn_z_adj = os.path.join(out_dir, os.path.splitext(slv_dem)[0][:-2] + '_Z_adj.tif')
        fn_corr_adj = os.path.join(out_dir, os.path.splitext(slv_dem)[0][:-2] + '_CORR_adj.tif')
        fn_v123_adj = os.path.join(out_dir, os.path.splitext(slv_dem)[0][:-2] + '_V123_adj.tif')

        list_fn_rm = [fn_slv_tif, fn_corr_tif, fn_along3N_tif, fn_along3B_tif, fn_v123_tif, fn_filtz, fn_z_adj,
                      fn_corr_adj, fn_v123_adj]

    mkdir_p(out_dir)

    # Prepare LOG files 
    if write_log:
        print(os.getcwd())
        logfile = open(os.path.join(out_dir, 'mmaster_bias_correct_' + str(os.getpid()) + '.log'), 'w')
        errfile = open(os.path.join(out_dir, 'mmaster_bias_correct_' + str(os.getpid()) + '_error.log'), 'w')
        sys.stdout = logfile
        sys.stderr = errfile

    # import angle data
    ang_mapN = GeoImg('TrackAngleMap_3N.tif')
    ang_mapB = GeoImg('TrackAngleMap_3B.tif')
    ang_mapNB = ang_mapN.copy(new_raster=np.array(np.divide(ang_mapN.img + ang_mapB.img, 2)))

    # pre-processing steps (co-registration,Correlation_masking)
    mst_coreg, slv_coreg, shift_params = preprocess(mst_dem, slv_dem, glacmask=glacmask, landmask=landmask,
                                                    work_dir='.', out_dir=out_dir, pts=pts)
    if shift_params == -1:
        print("Too few points for initial co-registration. Exiting.")
        clean_coreg_dir(out_dir, 'coreg')
        if write_log:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            logfile.close()
            errfile.close()
        if zipped:
            for fn_fil in list_fn_rm:
                if os.path.exists(fn_fil):
                    os.remove(fn_fil)
        if tmp_dir is not None:
            shutil.rmtree(os.path.join(proc_dir, work_dir))
        os.chdir(orig_dir)
        if return_geoimg:
            return slv_coreg, mst_coreg
        else:
            return
    clean_coreg_dir(out_dir, 'coreg')

    # OPEN and start the Results.pdf
    pp = PdfPages(os.path.sep.join([out_dir, 'BiasCorrections_Results.pdf']))

    ### Create the stable terrain masks
    stable_mask = create_stable_mask(slv_coreg, glacmask, landmask)
    fmaskpoly = get_aster_footprint(slv_dem.rsplit('_Z.tif')[0], proj4='+units=m +init=epsg:{}'.format(slv_coreg.epsg))
    fmask = make_mask(fmaskpoly, slv_coreg, raster_out=True)

    ### PREPARE numpy masked arrays for .img data
    smask = np.logical_or.reduce((np.invert(fmask), stable_mask, np.isnan(slv_coreg.img)))
    slv_coreg.unmask()
    slv_coreg.mask(smask)
    if pts:
        mst_coreg.clean()
        stable_mask = slv_coreg.copy(new_raster=smask)
        mst_coreg.mask(stable_mask.raster_points2(mst_coreg.xy) == 0)

    ### Create initial plot of where stable terrain is, including ICESat pts
    fig1 = plt.figure(figsize=(7, 5), facecolor='w', dpi=200)
    fig1, cimg = plot_shaded_dem(slv_coreg, fig=fig1)
    ax = fig1.gca()
    ax.set_title('Slave DEM Shaded Relief')
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')

    if pts:
        plt.plot(mst_coreg.x[~np.isnan(mst_coreg.elev)], mst_coreg.y[~np.isnan(mst_coreg.elev)], 'k.')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cimg, cax=cax)
    pp.savefig(fig1, dpi=200)

    ### cross-track bias removal 
    slv_coreg_xcorr, xcorr, pcoef = correct_cross_track_bias(mst_coreg, slv_coreg, ang_mapNB, pp, pts=pts)
    plt.close("all")

    ### along-track bias removal
    low_freq, all_freq = correct_along_track_bias(mst_coreg, slv_coreg_xcorr, ang_mapN, ang_mapB,
                                                  pp, pts=pts, robust=robust)
    slv_coreg_xcorr_acorr, acorr, scoef = low_freq
    slv_coreg_xcorr_acorr_jcorr, jcorr, jcoef = all_freq
    plt.close("all")

    ### Calculate dH and statistics    
    dH0 = calculate_dH(mst_coreg, slv_coreg, pts)
    dH1 = calculate_dH(mst_coreg, slv_coreg_xcorr, pts)
    dH2 = calculate_dH(mst_coreg, slv_coreg_xcorr_acorr, pts)
    dH_final = calculate_dH(mst_coreg, slv_coreg_xcorr_acorr_jcorr, pts)

    ### mask dH for 
    if not pts:
        # Calculate initial differences
        mytitle = 'dH Initial'
        false_hillshade(dH0, mytitle, pp=pp)

        # Calculate After Cross Track Changes
        mytitle = 'dH After Cross Track Corrections'
        false_hillshade(dH1, mytitle, pp=pp)

        # Calculate After Cross Track Changes
        mytitle = 'dH After Low Frequency Along Track Corrections'
        false_hillshade(dH2, mytitle, pp=pp)

        # Calculate After Cross Track Changes - TWICE TO SHOW THE ENHANCED SCALE
        mytitle = 'dH After Low Frequency Along Track Corrections'
        false_hillshade(dH2, mytitle, pp=pp, clim=(-7, 7))

        # Calculate post correction differences
        mytitle = 'dH After ALL Along Track Corrections'
        false_hillshade(dH_final, mytitle, pp=pp, clim=(-7, 7))

        final_histogram(dH0.img, dH1.img, dH2.img, dH_final.img, pp)
    elif pts:
        # Calculate initial differences
        final_histogram(dH0, dH1, dH2, dH_final, pp)
    #        final_histogram(dH0, dH1, dH_final, pp)

    #### PREPARE OUTPUT - have to apply the corrections to the original, unfiltered slave DEM.
    # first, we apply the co-registration shift.
    orig_slv = GeoImg(slv_dem)
    orig_slv.shift(shift_params[0], shift_params[1])
    orig_slv.img = orig_slv.img + shift_params[2]

    # now, calculate and apply the cross-track correction
    myang = np.deg2rad(ang_mapNB.img)
    xxr, _ = get_xy_rot(orig_slv, myang)
    cross_correction = fitfun_polynomial(xxr, pcoef)
    orig_slv.img = orig_slv.img + cross_correction

    outname = os.path.splitext(slv_dem)[0] + "_adj_X.tif"
    if not zipped:
        orig_slv.write(outname, out_folder=out_dir)
    np.savetxt(os.path.sep.join([out_dir, 'params_CrossTrack_Polynomial.txt']), pcoef)
    plt.close("all")

    # now, calculate and apply the along-track corrections
    xxn_mat, xxb_mat = get_atrack_coord(orig_slv, ang_mapN, ang_mapB)
    sinmod_low = fitfun_sumofsin_2angle(xxn_mat, xxb_mat, scoef)
    along_correction_low = np.reshape(sinmod_low, orig_slv.img.shape)
    orig_slv.img = orig_slv.img + along_correction_low

    outname = os.path.splitext(slv_dem)[0] + "_adj_XA.tif"
    if not zipped:
        orig_slv.write(outname, out_folder=out_dir)
    np.savetxt(os.path.sep.join([out_dir, 'params_AlongTrack_SumofSines.txt']), scoef)
    plt.close("all")

    # finally, calculate and apply the full-frequency along-track correction.
    sinmod = fitfun_sumofsin_2angle(xxn_mat, xxb_mat, jcoef)
    along_correction = np.reshape(sinmod, orig_slv.img.shape)
    # don't need to remove low freq correction with the new jitter approach
    orig_slv.img = orig_slv.img + along_correction

    outname = os.path.splitext(slv_dem)[0] + "_adj_XAJ.tif"
    if not zipped:
        orig_slv.write(outname, out_folder=out_dir)
    np.savetxt(os.path.sep.join([out_dir, 'params_AlongTrack_Jitter.txt']), jcoef)
    plt.close("all")

    pp.close()

    ### re-coregister
    print('Re-co-registering DEMs.')
    recoreg_outdir = os.path.sep.join([out_dir, 're-coreg'])
    mst_coreg, slv_adj_coreg, shift_params2 = dem_coregistration(mst_dem, slv_coreg_xcorr_acorr_jcorr,
                                                                 glaciermask=glacmask, landmask=landmask,
                                                                 outdir=recoreg_outdir, pts=pts)[0:3]
    if shift_params2 == -1:
        print("Too few points for final co-registration. Exiting.")
        clean_coreg_dir(out_dir, 're-coreg')
        if write_log:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            logfile.close()
            errfile.close()
        if zipped:
            for fn_fil in list_fn_rm:
                if os.path.exists(fn_fil):
                    os.remove(fn_fil)
        if tmp_dir is not None:
            shutil.rmtree(os.path.join(proc_dir, work_dir))
        os.chdir(orig_dir)
        if return_geoimg:
            return slv_coreg, mst_coreg
        else:
            return

    #write final outputs
    ast_name = slv_dem.rsplit('_Z.tif')[0]
    fn_z_final = os.path.join(out_dir,ast_name + '_Z_adj_XAJ_final.tif')
    fn_corr_final = os.path.join(out_dir,ast_name + '_CORR_adj_final.tif')
    fn_v123_final = os.path.join(out_dir, ast_name + '_V123_adj_final.tif')

    clean_coreg_dir(out_dir, 're-coreg')
    orig_slv.shift(shift_params2[0], shift_params2[1])
    orig_slv.img = orig_slv.img + shift_params2[2]
    orig_slv.write(os.path.basename(fn_z_final), out_folder=out_dir)

    corr = GeoImg(ast_name + '_CORR.tif')
    corr.shift(shift_params2[0], shift_params2[1])
    corr.write(os.path.basename(fn_corr_final), out_folder=out_dir, dtype=np.uint8)

    #commented writing of V123 for disk usage
    # ortho = GeoImg(ast_name + '_V123.tif')
    # ortho.shift(shift_params2[0], shift_params2[1])
    # ortho.write(os.path.basename(fn_v123_final), out_folder=out_dir, dtype=np.uint8)

    plt.close("all")
    # clean-up 
    print("Fin. Final. Finale.")

    if write_log:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        logfile.close()
        errfile.close()

    if zipped:
        #we want to copy metadata here, easier to get image extent when zipped...
        list_fn_met = glob(os.path.join(proc_dir, work_dir,'*.met'))
        for fn_met in list_fn_met:
            shutil.copy(fn_met,out_dir)
        #create zip archive with existing output files
        fn_zip_final = os.path.join(out_dir,ast_name+'_final.zip')
        list_fn_tozip = [fn_z_final,fn_corr_final,fn_v123_final]
        for fn_tozip in list_fn_tozip:
            if not os.path.exists(fn_tozip):
                list_fn_tozip.remove(fn_tozip)
        if len(list_fn_tozip)>0:
            create_zip_from_flist(list_fn_tozip,fn_zip_final)
        # remove all files but final zip file and stats/metadata
        list_fn_rm += list_fn_tozip
        for fn_fil in list_fn_rm:
            if os.path.exists(fn_fil):
                os.remove(fn_fil)

    if tmp_dir is not None:
        if final_dir is not None:
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            mkdir_p(os.path.basename(final_dir))
            shutil.move(out_dir,final_dir)
        shutil.rmtree(os.path.join(proc_dir,work_dir))

    os.chdir(orig_dir)
    if return_geoimg:
        return slv_adj_coreg, mst_coreg
