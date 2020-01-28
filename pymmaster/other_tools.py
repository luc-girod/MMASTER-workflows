from __future__ import print_function
import gdal
import osr
import ogr
import gdalconst
import os
import sys
import numpy as np
import pymmaster.fit_tools as ft
import math as m
import pandas as pd
from scipy import integrate
from datetime import datetime
import functools

#mix of methods from my own libs, we'll sort later

# AST L1A METHODS
def extract_odl_astL1A(fn):
    f = open(fn, 'r')
    body = f.read()

    def get_odl_parenth_value(text_odl, obj_name):
        posobj = str.find(text_odl, obj_name)
        posval = str.find(text_odl[posobj + 1:len(text_odl)], 'VALUE')
        posparenthesis = str.find(text_odl[posobj + 1 + posval:len(text_odl)], '(')
        posendval = str.find(text_odl[posobj + 1 + posval + posparenthesis:len(text_odl)], ')')

        val = text_odl[posobj + posval + posparenthesis + 2:posobj + posval + posparenthesis + posendval + 1]

        return val

    def get_odl_quot_value(text_odl, obj_name):
        posobj = str.find(text_odl, obj_name)
        posval = str.find(text_odl[posobj + 1:len(text_odl)], 'VALUE')
        posquote = str.find(text_odl[posobj + 1 + posval:len(text_odl)], '"')
        posendval = str.find(text_odl[posobj + posval + posquote + 2:len(text_odl)], '"')

        val = text_odl[posobj + posval + posquote + 2:posobj + posval + +posquote + posendval + 2]

        return val

    # get latitude
    lat_val = get_odl_parenth_value(body, 'GRingPointLatitude')
    lat_tuple = [float(lat_val.split(',')[0]), float(lat_val.split(',')[1]), float(lat_val.split(',')[2]),
                 float(lat_val.split(',')[3])]

    # get longitude
    lon_val = get_odl_parenth_value(body, 'GRingPointLongitude')
    lon_tuple = [float(lon_val.split(',')[0]), float(lon_val.split(',')[1]), float(lon_val.split(',')[2]),
                 float(lon_val.split(',')[3])]

    # get calendar date + time of day
    caldat_val = get_odl_quot_value(body, 'CalendarDate')
    timeday_val = get_odl_quot_value(body, 'TimeofDay')
    caldat = datetime(year=int(caldat_val.split('-')[0]), month=int(caldat_val.split('-')[1]),
                      day=int(caldat_val.split('-')[2]),
                      hour=int(timeday_val.split(':')[0]), minute=int(timeday_val.split(':')[1]),
                      second=int(timeday_val.split(':')[2][0:2]),
                      microsecond=int(timeday_val.split(':')[2][3:6]) * 1000)

    # get cloud cover
    cloudcov_val = get_odl_quot_value(body, 'SceneCloudCoverage')
    cloudcov_perc = int(cloudcov_val)

    # get flag if bands acquired or not: band 1,2,3N,3B,4,5,6,7,8,9,10,11,12,13,14
    list_band = []
    band_attr = get_odl_quot_value(body, 'Band3N_Available')
    band_avail = band_attr[0:3] == 'Yes'
    list_band.append(band_avail)
    band_attr = get_odl_quot_value(body, 'Band3B_Available')
    band_avail = band_attr[0:3] == 'Yes'
    list_band.append(band_avail)

    range_band = list(range(1, 15))
    range_band.remove(3)
    for i in range_band:
        band_attr = get_odl_quot_value(body, 'Band' + str(i) + '_Available')
        band_avail = band_attr[0:3] == 'Yes'
        list_band.append(band_avail)

    band_tags = pd.DataFrame(data=list_band,
                             index=['band_3N', 'band_3B', 'band_1', 'band_2', 'band_4', 'band_5', 'band_6', 'band_7',
                                    'band_8', 'band_9', 'band_10', 'band_11', 'band_12', 'band_13', 'band_14'])

    # get scene orientation angle
    orient_attr = get_odl_quot_value(body, 'ASTERSceneOrientationAngle')
    # orient_angl = float(orient_attr)
    # some .met files are in fact somehow incomplete for angles... let's forget it!
    orient_angl = float(15.)

    return lat_tuple, lon_tuple, caldat, cloudcov_perc, band_tags, orient_angl


def l1astrip_polygon(l1a_subdir):
    # number of l1a granules
    strip_l1a = [os.path.join(l1a_subdir, l1a) for l1a in os.listdir(l1a_subdir) if l1a.endswith('.met')]

    list_poly = []
    for l1a in strip_l1a:
        lat_tup, lon_tup, _, _, _, _ = extract_odl_astL1A(l1a)

        max_lon = np.max(lon_tup)
        min_lon = np.min(lon_tup)

        if min_lon < -160 and max_lon > 160:
            # if this is happening, ladies and gentlemen, bad news, we definitely have an image on the dateline

            # let's do two full polygons from each side of the dateline...
            lon_rightside = np.array(lon_tup, dtype=float)
            lon_rightside[lon_rightside < -160] += 360

            lon_leftside = np.array(lon_tup, dtype=float)
            lon_leftside[lon_leftside > 160] -= 360

            rightside_coord = list(zip(list(lon_rightside) + [lon_rightside[0]], lat_tup + [lat_tup[0]]))
            rightside_poly = poly_from_coords(rightside_coord)

            leftside_coord = list(zip(list(lon_leftside) + [lon_leftside[0]], lat_tup + [lat_tup[0]]))
            leftside_poly = poly_from_coords(leftside_coord)

            # create a world polygon and get intersection
            world_coord = [(-180, -90), (-180, 90), (180, 90), (180, -90), (-180, -90)]
            world_poly = poly_from_coords(world_coord)

            leftside_inters = world_poly.Intersection(leftside_poly)
            rightside_inters = world_poly.Intersection(rightside_poly)

            # add both to list
            list_poly += [leftside_inters, rightside_inters]
        else:
            list_coord = list(zip(lon_tup + [lon_tup[0]], lat_tup + [lat_tup[0]]))
            poly = poly_from_coords(list_coord)
            list_poly.append(poly)

    multipoly = ogr.Geometry(ogr.wkbMultiPolygon)

    for i in range(len(list_poly)):
        # stacking polygons in multipolygon
        multipoly.AddGeometry(list_poly[i])

    cascadedpoly = multipoly.UnionCascaded()

    return cascadedpoly


## TILING AND VECTOR METHODS

def SRTMGL1_naming_to_latlon(tile_name):
    if tile_name[0] == 'S' or tile_name[0] == 's':
        lat = -int(tile_name[1:3])
    elif tile_name[0] == 'N' or tile_name[0] == 'n':
        lat = int(tile_name[1:3])
    else:
        sys.exit('Could not read latitude according to SRTMGL1 naming convention.')

    if tile_name[3] == 'W' or tile_name[3] == 'w':
        lon = -int(tile_name[4:7])
    elif tile_name[3] == 'E' or tile_name[3] == 'e':
        lon = int(tile_name[4:7])
    else:
        sys.exit('Could not read longitude according to SRTMGL1 naming convention.')

    return lat, lon

def latlon_to_SRTMGL1_naming(lat,lon):

    if lat<0:
        str_lat = 'S'
    else:
        str_lat = 'N'

    if lon<0:
        str_lon = 'W'
    else:
        str_lon = 'E'

    tile_name = str_lat+str(int(abs(np.floor(lat)))).zfill(2)+str_lon+str(int(abs(np.floor(lon)))).zfill(3)

    return tile_name


def latlon_to_UTM(lat,lon):
    # utm module excludes regions south of 80°S and north of 84°N, unpractical for global vector manipulation
    # utm_all = utm.from_latlon(lat,lon)
    # utm_nb=utm_all[2]

    # utm zone from longitude without exclusions
    if -180 <= lon < 180:
        utm_nb = int(
            np.floor((lon + 180) / 6)) + 1  # lon=-180 refers to UTM zone 1 towards East (West corner convention)
    else:
        sys.exit('Longitude value is out of range.')

    if 0 <= lat < 90:  # lat=0 refers to North (South corner convention)
        epsg = '326' + str(utm_nb).zfill(2)
        utm_zone = str(utm_nb).zfill(2) + 'N'
    elif -90 <= lat < 0:
        epsg = '327' + str(utm_nb).zfill(2)
        utm_zone = str(utm_nb).zfill(2) + 'S'
    else:
        sys.exit('Latitude value is out of range.')

    return epsg, utm_zone


def epsg_from_utm(utm_zone):
    str_utm_nb = utm_zone[0:2]
    str_utm_ns = utm_zone[2]

    if str_utm_ns == 'N':
        epsg = '326' + str_utm_nb
    elif str_utm_ns == 'S':
        epsg = '327' + str_utm_nb
    else:
        sys.exit('UTM format not recognized.')

    return int(epsg)


def utm_from_epsg(epsg):
    str_epsg = str(epsg)
    str_epsg_ns = str_epsg[0:3]
    str_epsg_nb = str_epsg[3:5]

    if str_epsg_ns == '326':
        utm = str_epsg_nb + 'N'
    elif str_epsg_ns == '327':
        utm = str_epsg_nb + 'S'
    else:
        sys.exit('EPSG UTM format not recognized.')

    return utm


def poly_utm_latlontile(tile_name, utm_zone):
    lat, lon = SRTMGL1_naming_to_latlon(tile_name)
    extent = lon, lat, lon + 1, lat + 1
    poly = poly_from_extent(extent)

    epsg_out = epsg_from_utm(utm_zone)  # tile can be projected in whatever utm zone
    trans = coord_trans(False, 4326, False, epsg_out)

    poly.Transform(trans)

    return poly


#create a regularly spaced extent in a utm zone, with 3 overlapping pixel on the edges for reprojections
def niceextent_utm_latlontile(tile_name, utm_zone, gsd):
    poly = poly_utm_latlontile(tile_name,utm_zone)
    xmin, ymin, xmax, ymax = extent_from_poly(poly)

    xmin = xmin - xmin % gsd - 3*gsd
    ymin = ymin - ymin % gsd - 3*gsd
    xmax = xmax - xmax % gsd + 3*gsd
    ymax = ymax - ymax % gsd + 3*gsd

    return xmin, ymin, xmax, ymax


def create_mem_shp(geom, srs, layer_name='NA', layer_type=ogr.wkbPolygon, field_id='ID', field_val='1'):
    ds = gdal.GetDriverByName('MEMORY').Create('test.shp', 0, 0, 0, gdal.OF_VECTOR)
    layer = ds.CreateLayer(layer_name, srs, layer_type)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn(field_id, ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # if there are multiple geometries, put the "for" loop here
    # create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField(field_id, field_val)

    # make geometry
    geom = ogr.CreateGeometryFromWkt(geom.ExportToWkt())
    feat.SetGeometry(geom)
    layer.CreateFeature(feat)
    layer = feat = geom = None

    return ds

def latlontile_nodatamask(geoimg,tile_name):

    #create latlon tile polygon in utm projection
    lat, lon = SRTMGL1_naming_to_latlon(tile_name)
    extent = lon, lat, lon + 1, lat + 1
    poly = poly_from_extent(extent)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    #put in a memory vector
    ds_shp = create_mem_shp(poly,srs)

    return geoimg_mask_on_feat_shp_ds(ds_shp, geoimg)

def poly_from_coords(list_coord):
    # creating granule polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in list_coord:
        ring.AddPoint(float(coord[0]), float(coord[1]))

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly

def get_poly_centroid(poly):
    centroid = poly.Centroid()

    center_lon, center_lat, _ = centroid.GetPoint()

    return center_lon, center_lat

def extent_rast(raster_in):
    ds = gdal.Open(raster_in, gdalconst.GA_ReadOnly)
    x0_ref, dx_ref, dxdy_ref, y0_ref, dydx_ref, dy_ref = ds.GetGeoTransform()
    proj_wkt = ds.GetProjection()
    col_tot = ds.RasterXSize
    lin_tot = ds.RasterYSize
    x1_ref = x0_ref + col_tot * dx_ref
    y1_ref = y0_ref + lin_tot * dy_ref
    ds = None

    # extent format: Xmin, Ymin, Xmax, Ymax
    xmin = min(x0_ref, x1_ref)
    ymin = min(y0_ref, y1_ref)
    xmax = max(x0_ref, x1_ref)
    ymax = max(y0_ref, y1_ref)

    extent = [xmin, ymin, xmax, ymax]

    return extent, proj_wkt


def poly_from_extent(extent):
    # create a polygon from extent, coordinates order as in gdal
    xmin, ymin, xmax, ymax = extent

    ring = ogr.Geometry(ogr.wkbLinearRing)  # creating polygon ring
    ring.AddPoint(xmin, ymin)
    ring.AddPoint(xmax, ymin)
    ring.AddPoint(xmax, ymax)
    ring.AddPoint(xmin, ymax)
    ring.AddPoint(xmin, ymin)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)  # creating polygon

    return poly


def extent_from_poly(poly):
    env = poly.GetEnvelope()
    extent = env[0], env[2], env[1], env[3]

    return extent


def coord_trans(is_src_wkt, proj_src, is_tgt_wkt, proj_tgt):
    # choice between WKT or EPSG
    source_proj = osr.SpatialReference()
    if is_src_wkt:
        source_proj.ImportFromWkt(proj_src)
    else:
        source_proj.ImportFromEPSG(proj_src)

    target_proj = osr.SpatialReference()
    if is_tgt_wkt:
        target_proj.ImportFromWkt(proj_tgt)
    else:
        target_proj.ImportFromEPSG(proj_tgt)

    transform = osr.CoordinateTransformation(source_proj, target_proj)

    return transform


def list_shp_field_inters_extent(fn_shp, field_name, extent, proj_ext):
    poly = poly_from_extent(extent)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(fn_shp, 0)
    layer = ds.GetLayer()

    proj_shp = layer.GetSpatialRef().ExportToWkt()

    trans = coord_trans(True, proj_ext, True, proj_shp)

    poly.Transform(trans)

    list_field_inters = []
    for feat in layer:
        feat_geom = feat.GetGeometryRef()
        # inters = feat_geom.Intersection(poly)

        if feat_geom.Intersect(poly):
            list_field_inters.append(feat.GetField(field_name))

    return list_field_inters

def inters_list_poly_with_poly(list_poly,poly):

    list_inters=[]
    for poly_2 in list_poly:
        inters = poly_2.Intersection(poly)

        if not inters.IsEmpty():
            list_inters.append(poly_2)

    return list_inters


def create_mem_raster_on_geoimg(geoimg):
    masktarget = gdal.GetDriverByName('MEM').Create('', geoimg.npix_x, geoimg.npix_y, 1, gdal.GDT_Byte)
    masktarget.SetGeoTransform((geoimg.xmin, geoimg.dx, 0, geoimg.ymax, 0, geoimg.dy))
    masktarget.SetProjection(geoimg.proj_wkt)
    masktarget.GetRasterBand(1).Fill(0)

    return masktarget


def geoimg_mask_on_feat_shp_ds(shp_ds, geoimg, layer_name='NA', feat_id='ID', feat_val='1', **kwargs):
    ds_out = create_mem_raster_on_geoimg(geoimg)
    rasterize_feat_shp_ds(shp_ds, ds_out, layer_name=layer_name, feat_id=feat_id, feat_val=feat_val, **kwargs)
    mask = ds_out.GetRasterBand(1).ReadAsArray()
    mask = mask.astype(float)

    return mask == 1


def rasterize_feat_shp_ds(shp_ds, raster_ds, layer_name='NA', feat_id='ID', feat_val='1', all_touched=False,
                          exclude=False):
    if not exclude:
        str_eq = "='"
    else:
        str_eq = "!='"

    sql_stat = 'SELECT * FROM ' + layer_name + ' WHERE ' + feat_id + str_eq + feat_val + "'"

    opts = gdal.RasterizeOptions(burnValues=[1], bands=[1], SQLStatement=sql_stat, allTouched=all_touched)
    gdal.Rasterize(raster_ds, shp_ds, options=opts)


##INTERP AND STATS METHODS

def idx_near_val(array, v):
    return np.nanargmin(np.abs(array - v))


def std_err_finite(std, Neff, neff):
    return std * np.sqrt(1 / Neff * (Neff - neff) / Neff)


def std_err(std, Neff):
    return std * np.sqrt(1 / Neff)


def linear_err(delta_x, std_acc_y):
    return delta_x ** 2 / 8 * std_acc_y


def gauss(n=11, sigma=1):
    r = range(-int(n / 2), int(n / 2) + 1)
    return [1 / (sigma * m.sqrt(2 * m.pi)) * m.exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]


def kernel_exclass(xi, x0, a1, kappa=0.5):
    return np.exp(-(np.abs(xi - x0) / a1) ** kappa)


def kernel_exp(xi, x0, a1):
    return np.exp(-np.abs(xi - x0) / a1)


def kernel_gaussian(xi, x0, a1):
    return np.exp(-((xi - x0) / a1) ** 2)


def kernel_sph(xi,x0,a1):
    if np.abs(xi - x0) > a1:
        return 0
    else:
        return 1 - 3 / 2 * np.abs(xi-x0) / a1 + 1 / 2 * (np.abs(xi-x0) / a1) ** 3


def lowess_homemade_kern(x, y, w, a1, kernel='Exp'):
    """
    #inspired by: https://xavierbourretsicotte.github.io/loess.html
    homebaked lowess with variogram kernel + heteroscedasticity of observations with error

    :param x:
    :param y:
    :param w: heteroscedastic weights (inverse of variance)
    :param a1: range of the kernel (in variogram terms)
    :param kernel: kernel function
    :return:
    """

    n = len(x)
    yest = np.zeros(n)
    err_yest = np.zeros(n)

    if kernel == 'Gau':
        kernel_fun = kernel_gaussian
    elif kernel == 'Exp':
        kernel_fun = kernel_exp
    elif kernel == 'Exc':
        kernel_fun = kernel_exclass
    else:
        print('Kernel not recognized.')
        sys.exit()

    W = np.array([kernel_fun(x, x[i], a1) * w for i in range(n)]).T
    X = np.array([x for i in range(n)]).T
    Y = np.array([y for i in range(n)]).T

    beta1, beta0, _, Yl, Yu = ft.wls_matrix(X, Y, W, conf_interv=0.68)

    for i in range(n):
        yest[i] = beta1[i] * x[i] + beta0[i]
        err_yest[i] = (Yu[i, i] - Yl[i, i]) / 2

    return yest, err_yest


def interp_linear(xp, yp, errp, acc_y, loo=False):
    # interpolation 1d with error propagation: nan are considered void, possible leave-one-out (reinterpolate each value)

    # getting void index
    idx_void = np.isnan(yp)

    # preallocating arrays
    yp_out = np.copy(yp)
    errp_out = np.copy(errp)
    errlin_out = np.zeros(len(yp)) * np.nan

    # don't really care about performance, it's fast anyway, so let's do this one at a time
    for i in np.arange(len(xp)):

        x0 = xp[i]
        tmp_xp = np.copy(xp)
        tmp_xp[idx_void] = np.nan
        if loo:
            tmp_xp[i] = np.nan  # this is for leave-one out
        else:
            if not np.isnan(tmp_xp[i]):
                continue

        # find closest non void bin
        idx_1 = idx_near_val(tmp_xp, x0)
        tmp_xp[idx_1] = np.nan
        # second closest
        idx_2 = idx_near_val(tmp_xp, x0)

        # linear interpolation (or extrapolation)
        a = (xp[idx_2] - x0) / (xp[idx_2] - xp[idx_1])
        b = (x0 - xp[idx_1]) / (xp[idx_2] - xp[idx_1])

        # propagating standard error
        y0_out = a * yp[idx_1] + b * yp[idx_2]
        err0_out = np.sqrt(a ** 2 * errp[idx_1] ** 2 + b ** 2 * errp[idx_2] ** 2)
        # err0_out = np.sqrt(errp[idx_1]**2 + errp[idx_2]**2)

        # estimating linear error
        delta_x = min(np.absolute(xp[idx_2] - x0), np.absolute(xp[idx_1] - x0))
        errlin0_out = linear_err(delta_x, acc_y)

        # appending
        yp_out[i] = y0_out
        errp_out[i] = err0_out
        errlin_out[i] = errlin0_out

    return yp_out, errp_out, errlin_out


def interp_lowess(xp, yp, errp, acc_y, rang, kernel='Exc'):
    # interp1d with local regression and error propagation

    yp_out, errp_out = lowess_homemade_kern(xp, yp, 1 / (errp ** 2), a1=rang / 4., kernel=kernel)

    idx_void = np.isnan(yp)
    errlin_out = np.zeros(len(yp)) * np.nan

    for i in np.arange(len(xp)):

        x0 = xp[i]
        tmp_xp = np.copy(xp)
        tmp_xp[idx_void] = np.nan
        if not np.isnan(tmp_xp[i]):
            continue

        # find closest non void bin
        idx_1 = idx_near_val(tmp_xp, x0)
        tmp_xp[idx_1] = np.nan

        delta_x = np.absolute(xp[idx_1] - x0)
        errlin0_out = linear_err(delta_x, acc_y)
        errlin_out[i] = np.sqrt(errlin0_out ** 2 + errp[idx_1] ** 2)

    return yp_out, errp_out, errlin_out


def neff_sphsum_circular(area,crange1,psill1,crange2,psill2):

    #short range variogram
    c1 = psill1 # partial sill
    a1 = crange1  # short correlation range

    #long range variogram
    c1_2 = psill2
    a1_2 = crange2 # long correlation range

    h_equiv = np.sqrt(area / np.pi)

    #hypothesis of a circular shape to integrate variogram model
    if h_equiv > a1_2:
        std_err = np.sqrt(c1 * a1 ** 2 / (5 * h_equiv ** 2) + c1_2 * a1_2 ** 2 / (5 * h_equiv ** 2))
    elif (h_equiv < a1_2) and (h_equiv > a1):
        std_err = np.sqrt(c1 * a1 ** 2 / (5 * h_equiv ** 2) + c1_2 * (1-h_equiv / a1_2+1 / 5 * (h_equiv / a1_2) ** 3))
    else:
        std_err = np.sqrt(c1 * (1-h_equiv / a1+1 / 5 * (h_equiv / a1) ** 3) + c1_2 * (1-h_equiv / a1_2+1 / 5 * (h_equiv / a1_2) ** 3))

    return (psill1 + psill2)/std_err**2

def neff_circ(area,list_vgm):

    psill_tot = 0
    for vario in list_vgm:
        psill_tot += vario[2]

    def hcov_sum(h):
        fn = 0
        for vario in list_vgm:
            crange, model, psill = vario
            fn += h*(cov(h,crange,model=model,psill=psill))

        return fn

    h_equiv = np.sqrt(area / np.pi)

    full_int = integrate_fun(hcov_sum,0,h_equiv)[0]
    std_err = np.sqrt(2*np.pi*full_int / area)

    return psill_tot/std_err**2


def neff_rect(area,width,crange1,psill1,model1='Sph',crange2=None,psill2=None,model2=None):

    def hcov_sum(h,crange1=crange1,psill1=psill1,model1=model1,crange2=crange2,psill2=psill2,model2=model2):

        if crange2 is None or psill2 is None or model2 is None:
            return h*(cov(h,crange1,model=model1,psill=psill1))
        else:
            return h*(cov(h,crange1,model=model1,psill=psill1)+cov(h,crange2,model=model2,psill=psill2))

    width = min(width,area/width)

    full_int = integrate_fun(hcov_sum,0,width/2)[0]
    bin_int = np.linspace(width/2,area/width,100)
    for i in range(len(bin_int)-1):
        low = bin_int[i]
        upp = bin_int[i+1]
        mid = bin_int[i] + (bin_int[i+1]- bin_int[i])/2
        piec_int = integrate_fun(hcov_sum, low, upp)[0]
        full_int += piec_int * 2/np.pi*np.arctan(width/(2*mid))

    std_err = np.sqrt(2*np.pi*full_int / area)

    if crange2 is None or psill2 is None or model2 is None:
        return psill1 / std_err ** 2
    else:
        return (psill1 + psill2) / std_err ** 2


def integrate_fun(fun,low_limit,up_limit):

    return integrate.quad(fun,low_limit,up_limit)

def cov(h,crange,model='Sph',psill=1.,kappa=1/2,nugget=0):

    return (nugget + psill) - vgm(h,crange,model=model,psill=psill,kappa=kappa)

def vgm(h,crange,model='Sph',psill=1.,kappa=1/2,nugget=0):

    c0 = nugget #nugget
    c1 = psill #partial sill
    a1 = crange #correlation range
    s = kappa #smoothness parameter for Matern class

    if model == 'Sph':  # spherical model
        if h < a1:
            vgm = c0 + c1 * (3 / 2 * h / a1-1 / 2 * (h / a1) ** 3)
        else:
            vgm = c0 + c1
    elif model == 'Exp':  # exponential model
        vgm = c0 + c1 * (1-np.exp(-h / a1))
    elif model == 'Gau':  # gaussian model
        vgm = c0 + c1 * (1-np.exp(- (h / a1) ** 2))
    elif model == 'Exc':  # stable exponential model
        vgm = c0 + c1 * (1-np.exp(-(h/ a1)**s))

    return vgm

def double_sum_covar(tot_err, slope_bin, elev_bin, area_tot, crange, kernel):
    n = len(tot_err)

    dist_bin = np.zeros(n)
    # change elev binning in distances:
    bin_size = elev_bin[1] - elev_bin[0]

    for i in range(n - 1):
        ind = elev_bin <= elev_bin[i]
        tmpslope = slope_bin[ind]

        dist_bin[i + 1] = np.nansum(bin_size / np.tan(tmpslope * np.pi / 180.))

    var_err = 0
    for i in range(n):
        for j in range(n):
            var_err += kernel(dist_bin[i], dist_bin[j], crange) * tot_err[i] * tot_err[j] * area_tot[i] * area_tot[j]

    var_err /= np.nansum(area_tot) ** 2

    return np.sqrt(var_err)


def hypso_dc(dh_dc, err_dc, ref_elev, dt, tvals, mask, gsd, slope=None, bin_type='fixed', bin_val=100., filt_bin='3NMAD', method='linear'):

    valid_points = np.count_nonzero(np.logical_and(np.logical_and(~np.isnan(ref_elev),~np.isnan(dh_dc[0,:])),mask))

    #closest observation binning: monthly
    dt_bin = np.arange(0,np.nanmax(np.abs(dt))+30,5)
    nb_dt_bin = len(dt_bin)-1

    #amplitude of correlation length, calibrated with ICESat global
    corr_ranges = [500,5000,50000]
    corr_a = [25,35,35]
    corr_b = [0.2,0.02,0.0002]
    corr_c = [0.25,0.5,0.85]
    def sill_frac(t,a,b,c):
        return a*(1 - np.exp(-b*t**c))
    corr_std_dt = [functools.partial(sill_frac,a=corr_a[i],b=corr_b[i],c=corr_c[i]) for i in range(len(corr_ranges))]

    if valid_points == 0:
        df = pd.DataFrame()
        df = df.assign(hypso=np.nan, time=np.nan, dh=np.nan, err_dh=np.nan)
        df_hyp = pd.DataFrame()
        df_hyp = df_hyp.assign(hypso=np.nan, area_meas=np.nan, area_tot=np.nan, nmad=np.nan)
        df_int = pd.DataFrame()
        df_int = df_int.assign(time=tvals, dh=np.nan, err_dh=np.nan, area=np.nan)

        return df, df_hyp, df_int

    #elevation binning
    min_elev = np.nanmin(ref_elev[mask]) - (np.nanmin(ref_elev[mask]) % bin_val)
    max_elev = np.nanmax(ref_elev[mask])

    if max_elev == min_elev:
        max_elev = min_elev + 0.00001
    if bin_type == 'fixed':
        bin_final = bin_val
    elif bin_type == 'percentage':
        bin_final = np.ceil(bin_val / 100. * (max_elev - min_elev))
    else:
        sys.exit('Bin type not recognized.')
    bins_on_mask = np.arange(min_elev, max_elev, bin_final)
    nb_bin = len(bins_on_mask)

    #index only glacier pixels
    ref_on_mask = ref_elev[mask]
    dh_on_mask = dh_dc[:, mask]
    err_on_mask = err_dc[:, mask]
    dt_on_mask=dt[:,mask]

    #local hypsometric method (McNabb et al., 2019)

    #preallocating
    elev_bin, slope_bin, area_tot_bin, area_meas_bin, nmad_bin, std_geo_err = (np.zeros(nb_bin) * np.nan for i in range(6))
    mean_bin, std_bin, ss_err_bin, std_num_err, std_all_err = (np.zeros((nb_bin, np.shape(dh_dc)[0])) * np.nan for i in range(5))
    final_num_err_corr = np.zeros((np.shape(dh_dc)[0],len(corr_ranges)+1))*np.nan

    for i in np.arange(nb_bin):

        idx_bin = np.array(ref_on_mask >= bins_on_mask[i]) & np.array(
            ref_on_mask < (bins_on_mask[i] + bin_final))
        idx_orig = np.array(ref_elev >= bins_on_mask[i]) & np.array(
            ref_elev < (bins_on_mask[i] + bin_final)) & mask
        area_tot_bin[i] = np.count_nonzero(idx_orig) * gsd ** 2
        area_meas_bin[i] = np.count_nonzero(idx_bin) * gsd ** 2
        elev_bin[i] = bins_on_mask[i] + bin_final / 2.
        if slope is None:
            slope_bin[i] = 30. #average slope
        else:
            slope_bin[i] = np.nanmean(slope[idx_orig])

        dh_bin = dh_on_mask[:, idx_bin]
        err_bin = err_on_mask[:, idx_bin]

        # with current fit, a value can only be NaN along the entire temporal axis
        nvalid = np.count_nonzero(~np.isnan(dh_bin[0, :]))

        if nvalid > 0:

            if filt_bin == '3NMAD' and nvalid > 10:

                # this is for a temporally varying NMAD, which doesn't seem to always be relevant...
                # need to change preallocation if using this one
                #     mad = np.nanmedian(np.absolute(dh_bin - med_bin[i, :, None]), axis=1)
                #     nmad_bin[i, :] = 1.4826 * mad
                #     idx_outlier = np.absolute(dh_bin - med_bin[i, :, None]) > 3 * nmad_bin[i, :, None]
                #     dh_bin[idx_outlier] = np.nan

                # this is for a fixed NMAD using only the max dh
                dh_tot = dh_bin[-1,:]
                med_tot = np.nanmedian(dh_bin)
                mad = np.nanmedian(np.absolute(dh_bin) - med_tot)
                nmad_bin[i] = 1.4826*mad
                idx_outlier = np.absolute(dh_tot - med_tot) > 3 * nmad_bin[i]
                nb_outlier = np.count_nonzero(idx_outlier)
                dh_bin[:,idx_outlier] = np.nan
                area_meas_bin[i] -= nb_outlier * gsd ** 2

                # ref_elev_out[idx_orig & np.array(np.absolute(ref_elev_out - med_bin[i]) > 3 * nmad)] = np.nan

            std_bin[i, :] = np.nanstd(dh_bin, axis=1)

            #normal mean
            # mean_bin[i,:] = np.nanmean(dh_bin,axis=1)

            # weighted mean
            weights = 1. / err_bin ** 2
            mean_bin[i, :] = np.nansum(dh_bin * weights, axis=1) / np.nansum(weights, axis=1)
            ss_err_bin[i, :] = np.sqrt(np.nanmean(err_bin**2, axis=1))

            # ref_elev_out[idx_orig & np.isnan(ref_elev_out)] = mean_bin[i]

    area_tot = np.nansum(area_tot_bin)
    area_meas = np.nansum(area_meas_bin)

    idx_nonvoid = area_meas_bin > 0

    #what kind of signal are we missing? this is dependent on the hypsometric elevation change correlation
    # over void areas, i.e. the size of the whole glacier
    crange_geo = 10**(2.+area_tot/10**6/2.5) #in meters, this is a decent empirical approximation of exponential hypsometric correlation length based on area size

    # what is the typical std of glacier elevation change: proportional to dh
    psill_bin = std_bin[:, -1]
    # psill_bin = mean_bin[:, -1]

    for i in range(len(area_tot_bin)):
        if idx_nonvoid[i]:
            Neff_geo = neff_rect(area_tot_bin[i],bin_final/(np.tan(slope_bin[i]*np.pi/180.)),crange_geo,psill_bin[i],model1='Exp')
            neff_geo = neff_rect(area_meas_bin[i],bin_final/(np.tan(slope_bin[i]*np.pi/180.)),crange_geo,psill_bin[i],model1='Exp')
            std_geo_err[i] = std_err_finite(psill_bin[i],Neff_geo,neff_geo)
        else:
            std_geo_err[i] = np.nan

    #what is our numerical error
    crange_num1 = 100 #ASTER base correlation length
    psill_num1 = 4**2 #ASTER base variance
    crange_num2 = 1500 #ASTER jitter correlation length
    psill_num2 = 0.6**2 #ASTER jitter has a mean amplitude of ~2m, let's say we corrected or filtered about 70%

    # first, get standard error for all non-void bins
    for i in range(len(area_tot_bin)):
        if idx_nonvoid[i]:
            neff = neff_rect(area_meas_bin[i],bin_final/(np.tan(slope_bin[i]*np.pi/180.)),crange1=crange_num1,psill1=psill_num1
                             ,model1='Sph',crange2=crange_num2,psill2=psill_num2,model2='Sph')
            std_num_err[i,:] = std_err(ss_err_bin[i,:], neff)
        else:
            std_num_err[i,:] = np.nan

    std_num_err[np.isnan(std_num_err)] = 0
    std_geo_err[np.isnan(std_geo_err)] = 0

    std_all_err[idx_nonvoid] = np.sqrt(std_num_err[idx_nonvoid] ** 2 + std_geo_err[idx_nonvoid,None] ** 2)

    # if method == 'linear':

        # # first, do a leave-one out linear interpolation to remove non-void bins with really low confidence
        # loo_mean, loo_std_err, loo_lin_err = interp_linear(elev_bin, mean_bin, std_all_err, acc_dh, loo=True)
        # loo_full_err = np.sqrt(loo_std_err ** 2 + loo_lin_err ** 2)
        #
        # idx_low_conf = nonvoid_err_bin > loo_full_err
        #
        # idx_final_void = np.logical_and(np.invert(idx_nonvoid), idx_low_conf)

        # then, interpolate for all of those bins
        # mean_bin[idx_final_void] = np.nan
        # nonvoid_err_bin[idx_final_void] = np.nan

        # final_std_err[~idx_final_void] = 0

    #linear interpolation of voids by temporal step
    # #TODO: optimize that temporally later

    final_mean_hyp, final_err_hyp = (np.zeros(np.shape(mean_bin))*np.nan for i in range(2))
    final_mean, final_err = (np.zeros(np.shape(mean_bin)[1])*np.nan for i in range(2))

    # without taking into account large scale spatial correlation
    # neff_num_tot = neff_circ(area_meas,crange1=crange_num1,psill1=psill_num1
    #                          ,model1='Sph',crange2=crange_num2,psill2=psill_num2,model2='Sph')

    intrabin_err = std_geo_err
    intrabin_err[np.isnan(intrabin_err)] = 0
    if nb_bin>1:
        final_geo_err = double_sum_covar(intrabin_err, slope_bin, elev_bin, area_tot_bin, crange_geo, kernel=kernel_exp)
    else:
        final_geo_err = intrabin_err[0]
    for i in range(np.shape(mean_bin)[1]):
        if nb_bin>1:
            tmp_mean, tmp_std_err, tmp_lin_err = interp_linear(elev_bin, mean_bin[:,i], std_all_err[:,i], acc_y=.002, loo=False)
        else:
            tmp_mean = np.array([mean_bin[0,i]])
            tmp_std_err = np.array([std_all_err[0,i]])
            tmp_lin_err = np.array([0])

        tmp_std_err[np.isnan(tmp_std_err)] = 0
        tmp_lin_err[np.isnan(tmp_lin_err)] = 0
        interbin_err = np.sqrt(tmp_std_err ** 2 + tmp_lin_err ** 2)
        tmp_tot_err = np.sqrt(interbin_err ** 2 + intrabin_err ** 2)

        final_mean_hyp[:, i] = tmp_mean
        final_err_hyp[:, i] = tmp_tot_err

        # integrate along hypsometry
        final_mean[i] = np.nansum(tmp_mean * area_tot_bin)/area_tot

        #without taking account a very large spatial correlation at the regional scale, only one error:
        # final_num_err = std_err(np.nansum(ss_err_bin[:,i]*area_meas_bin)/area_meas,neff_num_tot)


        nsamp_dt = np.zeros(nb_dt_bin)*np.nan
        err_corr = np.zeros((nb_dt_bin,len(corr_ranges)+1)) * np.nan

        for j in np.arange(nb_dt_bin):

            idx_dt_bin= np.logical_and(np.abs(dt_on_mask[i,:]) >= dt_bin[j],np.abs(dt_on_mask[i,:])<dt_bin[j+1])

            err_dt = err_on_mask[i,idx_dt_bin]

            final_num_err_dt=np.sqrt(np.nanmean(err_dt**2))
            nsamp_dt[j] = np.count_nonzero(idx_dt_bin)

            sum_var = 0
            for k in range(len(corr_ranges)+1):

                if k != len(corr_ranges):
                    err_corr[j, k] = np.sqrt(max(0,corr_std_dt[len(corr_ranges)-1-k](dt_bin[j] + (dt_bin[j + 1] - dt_bin[j]) / 2) - sum_var))
                    sum_var += err_corr[j, k] ** 2
                else:
                    err_corr[j, k]=np.sqrt(max(0,final_num_err_dt**2-sum_var))


        for k in range(len(corr_ranges)+1):
            final_num_err_corr[i,k] = np.sqrt(np.nansum(err_corr[:,k]*nsamp_dt)/np.nansum(nsamp_dt))

        list_vgm = [(corr_ranges[0],'Sph',final_num_err_corr[i,3]**2),(corr_ranges[1],'Sph',final_num_err_corr[i,2]**2),
                    (corr_ranges[2],'Sph',final_num_err_corr[i,1]**2),(500000,'Sph',final_num_err_corr[i,0]**2)]
        neff_num_tot = neff_circ(area_meas,list_vgm)

        final_num_err = std_err(np.nansum(final_num_err_corr[i,:]),neff_num_tot)

        final_err[i] = np.sqrt(final_num_err**2+final_geo_err**2)

    mean_dt = np.nanmean(np.abs(dt_on_mask),axis=1)

    #prepare index to write results
    hypso_index = np.array([h for h in elev_bin for t in tvals])
    time_index = np.array([t for h in elev_bin for t in tvals])

    #dataframe with hypsometric mean and error for all time steps
    df = pd.DataFrame()
    df = df.assign(hypso=hypso_index, time=time_index, dh=final_mean_hyp.flatten(), err_dh=final_err_hyp.flatten())

    #dataframe with hypsometric data
    df_hyp = pd.DataFrame()
    df_hyp = df_hyp.assign(hypso=elev_bin,area_meas=area_meas_bin,area_tot=area_tot_bin,nmad=nmad_bin)

    #dataframe with spatially integrated volume for all time steps
    df_int = pd.DataFrame()
    df_int = df_int.assign(time=tvals,dh=final_mean,err_dh=final_err,area=np.repeat(area_tot,len(tvals))
                           ,sill_corr05=final_num_err_corr[:,3],sill_corr5=final_num_err_corr[:,2],
                           sill_corr50=final_num_err_corr[:,1], sill_corr500=final_num_err_corr[:,0],dt=mean_dt)

    # for i in np.arange(nb_bin):
    #     idx_orig = np.array(ref_elev >= bins_on_mask[i]) & np.array(
    #         ref_elev < (bins_on_mask[i] + bin_final)) & mask
    #     if not idx_nonvoid[i]:
    #         ref_elev_out[idx_orig] = final_mean[i]

    # return df, dc_out

    return df, df_hyp, df_int
