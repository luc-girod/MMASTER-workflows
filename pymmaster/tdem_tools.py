"""
pymmaster.tdem_tools provides tools to post-process DEM stacks (volume integration, etc...)
"""
from __future__ import print_function
import xarray as xr
import os
import sys
import shutil
import numpy as np
from itertools import groupby
from operator import itemgetter
import multiprocessing as mp
from scipy.interpolate import interp1d
import gdal
import osr
import ogr
import time
from datetime import datetime as dt
import pandas as pd
import pymmaster.stack_tools as st
import pymmaster.other_tools as ot
import pymmaster.fit_tools as ft
from pybob.coreg_tools import get_slope, create_stable_mask, dem_coregistration
from pybob.GeoImg import GeoImg
from pybob.ICESat import ICESat
from glob import glob

def inters_feat_shp_stacks(fn_shp, list_fn_stack, feat_field_name):
    # get intersecting rgiid for each stack extent
    list_list_rgiid = []
    for fn_stack in list_fn_stack:
        #TODO: temporary test
        # ds = xr.open_dataset(fn_stack)
        # extent, proj = st.extent_stack(ds)
        ds = xr.open_dataset(fn_stack)
        tile_name = st.tilename_stack(ds)
        lat, lon = ot.SRTMGL1_naming_to_latlon(tile_name)
        extent = [lon, lat, lon+1, lat+1]
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326)

        list_rgiid = ot.list_shp_field_inters_extent(fn_shp, feat_field_name, extent, proj.ExportToWkt())
        list_list_rgiid.append(list_rgiid)

    # get all rgiids intersecting all stacks without duplicates
    all_rgiid = []
    for list_rgiid in list_list_rgiid:
        all_rgiid = all_rgiid + list_rgiid
    all_rgiid = list(set(all_rgiid))

    # inverting to have intersecting stacks by rgiid: more practical to compute that way
    list_list_stack_by_rgiid = []
    for rgiid in all_rgiid:
        list_stacks = []
        for fn_stack in list_fn_stack:
            if rgiid in list_list_rgiid[list_fn_stack.index(fn_stack)]:
                list_stacks.append(fn_stack)
        list_list_stack_by_rgiid.append(list_stacks)

    return all_rgiid, list_list_stack_by_rgiid


def sel_dc(ds, tlim, mask):
    # select data cube on temporal and spatial mask
    if tlim is None:
        time1 = time2 = None
    else:
        time1, time2 = tlim

    index = np.where(mask)
    minx = ds.x[np.min(index[1])].values
    maxx = ds.x[np.max(index[1])].values
    miny = ds.y[np.min(index[0])].values
    maxy = ds.y[np.max(index[0])].values

    dc = ds.sel(dict(time=slice(time1, time2), x=slice(minx, maxx), y=slice(miny, maxy)))
    submask = mask[np.min(index[0]):np.max(index[0] + 1), np.min(index[1]):np.max(index[1] + 1)]

    return dc, submask, (slice(minx,maxx),slice(miny,maxy))

def int_dc(dc, mask, **kwargs):
    # integrate data cube over masked area
    dh = dc.variables['z'].values - dc.variables['z'].values[0]
    err = np.sqrt(dc.variables['z_ci'].values ** 2 + dc.variables['z_ci'].values[0] ** 2)
    ref_elev = dc.variables['z'].values[0]
    slope = get_slope(st.make_geoimg(dc, 0)).img
    slope[np.logical_or(~np.isfinite(slope), slope > 70)] = np.nan

    t, y, x = np.shape(dh)
    dx = np.round((dc.x.max().values - dc.x.min().values) / float(x))

    df = ot.hypso_dc(dh,err,ref_elev,mask,dx,**kwargs)

    return df

def get_inters_stack_dem(list_fn_stack,ext,proj):

    # get stacks that intersect the extent
    poly = ot.poly_from_extent(ext)

    list_inters_stack = []
    for fn_stack in list_fn_stack:
        ds = xr.open_dataset(fn_stack)
        ext_st, proj_st = st.extent_stack(ds)
        poly_st = ot.poly_from_extent(ext_st)
        trans = ot.coord_trans(True, proj_st, True, proj)
        poly_st.Transform(trans)

        if poly_st.Intersect(poly):
            list_inters_stack.append(fn_stack)

    return list_inters_stack

def comp_stacks_dem(list_fn_stack,fn_dem,inc_mask=None, exc_mask=None, get_timelapse_filtstack=False, outfile=None):

    #get list of intersecting stacks
    ext, proj = ot.extent_rast(fn_dem)
    list_inters_stack = get_inters_stack_dem(list_fn_stack,ext,proj)
    dem_full = GeoImg(fn_dem)

    list_dh = list_z_score = list_dt = []
    for fn_stack in list_inters_stack:

        ds = xr.open_dataset(fn_stack)
        ref_img = st.make_geoimg(ds)
        mask = create_stable_mask(ref_img,exc_mask,inc_mask)

        dem = dem_full.reproject(ref_img)

        x,y = np.where(mask)
        t = np.array([dem.datetime]*len(x))

        dem_sub = dem[x,y]
        comp = ds.interp(time=t,x=x,y=y)
        comp_h = comp.variables['z'].values
        comp_ci = comp.variables['z_ci'].values

        dh = dem_sub - comp_h
        z_score = dh / comp_ci

        list_dh.append(dh)
        list_z_score.append(z_score)

        if get_timelapse_filtstack:
            #here we assume filtered stack is stored at the same location
            fn_filt = os.path.join(os.path.dirname(fn_stack),os.path.basename(fn_stack).split('_')[0]+'_filtered.nc')
            ds_filt = xr.open_dataset(fn_filt)

            ds_near = ds_filt.isel(time=t,x=x,y=y,method='nearest')
            near_times = ds_near.time.values
            delta_t = near_times - t

            list_dt.append(delta_t)

    full_dh = np.concatenate(list_dh)
    full_z_score = np.concatenate(list_z_score)
    full_dt = np.concatenate(list_dt)

    df = pd.DataFrame()
    df = df.assign(dh=full_dh,z_score=full_z_score,dt=full_dt)

    if outfile is None:
        return df
    else:
        df.to_csv(outfile)

def datetime_to_yearfraction(date):

    #ref:https://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years
    def sinceEpoch(date): #returns seconds since epoch

        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction


def create_tmp_dir_for_outfile(file_out):

    tmp_dir = os.path.join(os.path.dirname(file_out), 'tmp_'+os.path.splitext(os.path.basename(file_out))[0]) + os.path.sep
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    return tmp_dir

def remove_tmp_dir_for_outfile(file_out):

    tmp_dir = os.path.join(os.path.dirname(file_out), 'tmp_'+os.path.splitext(os.path.basename(file_out))[0]) + os.path.sep
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir,ignore_errors=True)

def saga_aspect_slope_curvature(dem_in, topo_out, method_nb=5):
    """
    :param dem_in: dem input
    :param topo_out: raster out (3 bands: aspect, slope, curvature)
    :param method_nb: algorithm used, see function description
    :return:

    requirement: SAGA 2.X with X>3

    #ref:http://www.saga-gis.org/saga_tool_doc/2.2.3/ta_morphometry_0.html

    aspect, slope, curvature methods
    methods number
    [0] maximum slope (Travis et al. 1975)
    [1] maximum triangle slope (Tarboton 1997)
    [2] least squares fitted plane (Horn 1981, Costa-Cabral & Burgess 1996)
    [3] 6 parameter 2nd order polynom (Evans 1979)
    [4] 6 parameter 2nd order polynom (Heerdegen & Beran 1982)
    [5] 6 parameter 2nd order polynom (Bauer, Rohdenburg, Bork 1985)
    [6] 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
    [7] 10 parameter 3rd order polynom (Haralick 1983)

    unit slope 0=rad, 1=deg, 2=percent
    unit aspect 0=rad, 1=deg
    """

    tmp_dir = create_tmp_dir_for_outfile(topo_out)

    saga_elev = tmp_dir + 'elev_temp.sgrd'
    saga_slope = tmp_dir + 'slope_temp.sgrd'
    saga_aspect = tmp_dir + 'aspect_temp.sgrd'
    saga_max_curv = tmp_dir + 'max_curv_temp.sgrd'
    tif_slope = tmp_dir + 'slope_temp.tif'
    tif_aspect = tmp_dir + 'aspect_temp.tif'
    tif_max_curv = tmp_dir + 'max_curv_temp.tif'
    output_vrt = tmp_dir + 'stack.vrt'

    os.system('saga_cmd io_gdal 0 -GRIDS ' + saga_elev + ' -FILES ' + dem_in)
    os.system(
        'saga_cmd ta_morphometry 0 -ELEVATION ' + saga_elev + ' -SLOPE ' + saga_slope + ' -ASPECT ' + saga_aspect + ' -C_MAXI ' + saga_max_curv + ' -METHOD ' + str(
            method_nb) + ' -UNIT_SLOPE 1 -UNIT_ASPECT 1')
    os.system('saga_cmd io_gdal 2 -GRIDS ' + saga_slope + ' -FILE ' + tif_slope)
    os.system('saga_cmd io_gdal 2 -GRIDS ' + saga_aspect + ' -FILE ' + tif_aspect)
    os.system('saga_cmd io_gdal 2 -GRIDS ' + saga_max_curv + ' -FILE ' + tif_max_curv)

    os.system(
        'gdalbuildvrt -separate -overwrite ' + output_vrt + ' ' + tif_slope + ' ' + tif_aspect + ' ' + tif_max_curv)
    os.system('gdal_translate ' + output_vrt + ' ' + topo_out)

    remove_tmp_dir_for_outfile(topo_out)


def icesat_comp_wrapper(argsin):

    fn_stack,ice_coords,ice_latlon,ice_elev,ice_date,groups,dates,read_filt,inc_mask,exc_mask = argsin

    full_h, full_dh, full_z_score, full_dt, full_pos, full_slp, full_lat, full_lon, full_dh_ref, full_curv = (np.array([]) for i in range(10))
    full_time = np.array([],dtype='datetime64[D]')
    ds = xr.open_dataset(fn_stack)
    ds.load()
    tile_name = st.tilename_stack(ds)

    tmp_h = st.make_geoimg(ds)
    tmp_ci = st.make_geoimg(ds)
    tmp_slope = st.make_geoimg(ds)
    tmp_slope.img = ds.slope.values
    # tmp_slope.img = np.ones(np.shape(tmp_slope.img))*20

    ds_sub = ds.interp(time=dates)

    terrain_mask = None
    get_curv = True

    if exc_mask is not None:

        print('Tile '+tile_name+': deriving terrain exclusion mask...')
        #rasterizing mask and changing to float to use geoimg.raster_points2
        mask = ft.get_stack_mask(exc_mask,ds)
        tmp_exc_mask = st.make_geoimg(ds)
        tmp_exc_mask.img = np.zeros(np.shape(mask),dtype='float32')
        tmp_exc_mask.img[mask] = 2

        terrain_mask = tmp_exc_mask

    if inc_mask is not None:
        print('Tile ' + tile_name + ': deriving terrain inclusion mask...')
        # rasterizing mask and changing to float to use geoimg.raster_points2
        mask = ft.get_stack_mask(inc_mask, ds)

        if terrain_mask is not None:
            land_mask = np.logical_and(terrain_mask.img == 0, mask)
            terrain_mask.img[land_mask] = 1
        else:
            tmp_inc_mask = st.make_geoimg(ds)
            tmp_inc_mask.img = np.zeros(np.shape(mask),dtype='float32')
            tmp_inc_mask[mask] = 1

            terrain_mask = tmp_inc_mask

    fn_raw = os.path.join(os.path.dirname(fn_stack), os.path.basename(fn_stack).split('_')[0] + '.nc')
    ds_raw = xr.open_dataset(fn_raw)
    tmp_ref_h = st.make_geoimg(ds_raw,band=(slice(0,None),slice(0,None)),var='ref_z')
    tmp_ref_h = tmp_ref_h.reproject(tmp_h)

    #get curvature
    if get_curv:
        fn_tmp_ref = os.path.join(os.path.dirname(fn_stack),'tmp_ref_'+os.path.basename(fn_stack).split('_')[0]+'.tif')
        tmp_ref_h.write(fn_tmp_ref)
        fn_tmp_curv = os.path.join(os.path.dirname(fn_stack),'tmp_curv_'+os.path.basename(fn_stack).split('_')[0]+'.tif')
        saga_aspect_slope_curvature(fn_tmp_ref,fn_tmp_curv)
        ds_curv = gdal.Open(fn_tmp_curv,gdal.GA_ReadOnly)
        curv_arr = ds_curv.GetRasterBand(3).ReadAsArray()
        tmp_curv = GeoImg(fn_tmp_curv)
        tmp_curv.img = curv_arr
        ds_curv = None
        os.remove(fn_tmp_curv)
        os.remove(fn_tmp_ref)
    else:
        tmp_curv = st.make_geoimg(ds)
        tmp_curv.img = np.zeros(np.shape(tmp_curv.img),dtype='float32')

    if read_filt:
        print('Tile '+tile_name+': getting original data dates from filtered array...')

        #we read the boolean data cube indicating positions where original data was used
        tmp_dt = st.make_geoimg(ds)
        fn_filt = os.path.join(os.path.dirname(fn_stack), os.path.basename(fn_stack).split('_')[0] + '_filtered.nc')
        ds_filt = xr.open_dataset(fn_filt)
        ds_filt.load()

        #first, remove duplicate dates by merging boolean arrays for same dates
        t_vals = list(ds_filt.time.values)
        dates_rm_dupli = list(set(t_vals))
        ind_firstdate = []
        for i, date in enumerate(dates_rm_dupli):
            ind_firstdate.append(t_vals.index(date))
        ind_firstdate = sorted(ind_firstdate)
        ds_filt2 = ds_filt.isel(time=np.array(ind_firstdate))
        for i in range(len(dates_rm_dupli)):
            t_ind = (t_vals == dates_rm_dupli[i])
            if len(t_ind) > 1:
                ds_filt2.z.values[i, :] = np.any(~ds_filt.z[t_vals == dates_rm_dupli[i], :].values.astype(bool), axis=0)

            # getting time data as days since 2000
        y0 = np.datetime64('2000-01-01')
        ftime = ds_filt2.time.values
        ftime_delta = np.array([t - y0 for t in ftime])
        days = [td.astype('timedelta64[D]').astype(int) for td in ftime_delta]

        # reindex to get closest not-NaN time value of the date vector
        filt_arr = np.array(ds_filt2.z.values, dtype='float32')
        filt_arr[filt_arr == 0] = np.nan
        days = np.array(days)
        filt_arr = filt_arr * days[:, None, None]
        at_least_2 = np.count_nonzero(~np.isnan(filt_arr), axis=0) > 2
        filt_tmp = filt_arr[:, at_least_2]
        out_arr = np.copy(filt_tmp)
        for i in range(np.shape(filt_tmp)[1]):
            ind = ~np.isnan(filt_tmp[:, i])
            fn = interp1d(days[ind], filt_tmp[:, i][ind], kind='nearest', fill_value='extrapolate', assume_sorted=True)
            out_arr[:, i] = fn(days)
        filt_arr[:, at_least_2] = out_arr
        ds_filt2.z.values = filt_arr
        ds_filt_sub = ds_filt2.reindex(time=dates, method='nearest')


    for i, group in enumerate(groups):
        #keep only a campaign group
        pts_idx_dt = np.array(ice_date == group)
        date = dates[i]

        print('Tile '+tile_name+': calculating differences for the ' + str(
            np.count_nonzero(pts_idx_dt)) + ' points of campaign:' + str(date))

        subsamp_ice = [tup for i, tup in enumerate(ice_coords) if pts_idx_dt[i]]
        subsamp_latlon = [tup for i, tup in enumerate(ice_latlon) if pts_idx_dt[i]]

        tmp_h.img = ds_sub.z[i, :].values
        comp_pts_h = tmp_h.raster_points(subsamp_ice, nsize=3, mode='linear')

        tmp_ci.img = ds_sub.z_ci[i, :].values
        comp_pts_ci = tmp_ci.raster_points(subsamp_ice, nsize=3, mode='linear')

        comp_pts_slope = tmp_slope.raster_points(subsamp_ice, nsize=3, mode='linear')

        comp_pts_ref_h = tmp_ref_h.raster_points(subsamp_ice, nsize=3, mode='linear')

        comp_pts_curv = tmp_curv.raster_points(subsamp_ice,nsize=3,mode='linear')

        dh = ice_elev[pts_idx_dt] - comp_pts_h
        good_vals = np.isfinite(dh)
        dh = dh[good_vals]
        h = ice_elev[pts_idx_dt][good_vals]
        z_score = dh / comp_pts_ci[good_vals]
        slp = comp_pts_slope[good_vals]
        dh_ref = ice_elev[pts_idx_dt] - comp_pts_ref_h
        dh_ref = dh_ref[good_vals]
        curv = comp_pts_curv[good_vals]

        if read_filt:
            day_diff = (date - y0).astype('timedelta64[D]').astype(int)
            tmp_dt.img = ds_filt_sub.z[i, :].values - np.ones(np.shape(tmp_dt.img)) * day_diff
            comp_pts_dt = tmp_dt.raster_points(subsamp_ice, nsize=3, mode='mean')
            dt_out = comp_pts_dt[good_vals]
        else:
            dt_out = np.zeros(len(dh)) * np.nan

        if terrain_mask is not None:
            comp_pts_mask = terrain_mask.raster_points(subsamp_ice, nsize=5, mode='nearest')
            pos = comp_pts_mask[good_vals]
        else:
            pos = np.zeros(len(dh))


        good_subsamp_latlon = [tup for i, tup in enumerate(subsamp_latlon) if good_vals[i]]
        if len(good_subsamp_latlon) > 0:
            lon = np.array([tup[1] for i, tup in enumerate(good_subsamp_latlon)])
            lat = np.array([tup[0] for i, tup in enumerate(good_subsamp_latlon)])
        else:
            lat=np.array([])
            lon=np.array([])

        full_h = np.concatenate([full_h,h])
        full_dh = np.concatenate([full_dh, dh])
        full_z_score = np.concatenate([full_z_score, z_score])
        full_dt = np.concatenate([full_dt,dt_out])
        full_pos = np.concatenate([full_pos,pos])
        full_slp = np.concatenate([full_slp,slp])
        full_time = np.concatenate([full_time,np.array([date]*len(dh),dtype='datetime64[D]')])
        full_lat = np.concatenate([full_lat,lat])
        full_lon = np.concatenate([full_lon,lon])
        full_dh_ref = np.concatenate([full_dh_ref,dh_ref])
        full_curv = np.concatenate([full_curv,curv])


    return full_h, full_dh, full_z_score, full_dt, full_pos, full_slp, full_time, full_lon, full_lat, full_dh_ref, full_curv


def comp_stacks_icesat(list_fn_stack,fn_icesat,inc_mask=None,exc_mask=None,nproc=1,read_filt=False,shift=None):

    ice = ICESat(fn_icesat)
    el_limit = -200
    mykeep = ice.elev > el_limit

    #exception for North Asia subreg 3 ICESat file
    if os.path.basename(fn_icesat) == 'ICESat_10_3_rgi50_NorthAsia.h5':
        keep_west = np.logical_and(ice.lon<-172,ice.lon>-179.999)
        keep_east_1 = np.logical_and(ice.lon<96,ice.lon>88)
        keep_east_2 = np.logical_and(ice.lon<146.999,ice.lon>138)
        keep_northasia_3 = np.logical_or.reduce((keep_west,keep_east_1,keep_east_2))
        mykeep = np.logical_and(mykeep,keep_northasia_3)

    ice.x = ice.x[mykeep]
    ice.y = ice.y[mykeep]
    ice.lat = ice.lat[mykeep]
    ice.lon = ice.lon[mykeep]
    ice.elev = ice.elev[mykeep]
    ice.UTCTime = ice.UTCTime[mykeep]
    ice.xy = list(zip(ice.x, ice.y))

    #putting code above in ICESat.clean() breaks co-registration...
    # ice.clean(el_limit=-200)

    #get intersecting tiles
    bounds = ice.get_bounds()
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    proj_wkt = proj.ExportToWkt()
    list_inters = get_inters_stack_dem(list_fn_stack,[bounds[0],bounds[2],bounds[1],bounds[3]],proj_wkt)

    if len(list_inters) == 0:
        print('Found no intersection to ICESat file: '+fn_icesat)
        raise ValueError('No intersecting file!')

    #laser campaigns of ICESat
    laser_op_icesat = [(dt(2003, 2, 20), dt(2003, 3, 29)), (dt(2003, 9, 25), dt(2003, 11, 19)),
                       (dt(2004, 2, 17), dt(2004, 3, 21)),
                       (dt(2004, 5, 18), dt(2004, 6, 21)), (dt(2004, 10, 3), dt(2004, 11, 8)),
                       (dt(2005, 2, 17), dt(2005, 3, 24)),
                       (dt(2005, 5, 20), dt(2005, 6, 23)), (dt(2005, 10, 21), dt(2005, 11, 24)),
                       (dt(2006, 2, 22), dt(2006, 3, 28)),
                       (dt(2006, 5, 24), dt(2006, 6, 26)), (dt(2006, 10, 25), dt(2006, 11, 27)),
                       (dt(2007, 3, 12), dt(2007, 4, 14)),
                       (dt(2007, 10, 2), dt(2007, 11, 5)), (dt(2008, 2, 17), dt(2008, 3, 21)),
                       (dt(2008, 10, 4), dt(2008, 10, 19)),
                       (dt(2008, 11, 25), dt(2008, 12, 17)), (dt(2009, 3, 9), dt(2009, 4, 11)),
                       (dt(2009, 9, 30), dt(2009, 10, 11))]
    #campaign names
    laser_op_name = ['1AB', '2A', '2B', '2C', '3A', '3B', '3C', '3D', '3E', '3F', '3G', '3H', '3I', '3J', '3K', '2D',
                     '2E', '2F']

    # group ICESat dates by operation
    utc_days = ice.UTCTime

    for i in range(len(laser_op_icesat)):
        # datetime to UTC days
        start_dt = datetime_to_yearfraction(laser_op_icesat[i][0]) * 365.2422 - 10
        end_dt = datetime_to_yearfraction(laser_op_icesat[i][1]) * 365.2422 + 10

        mean_dt = 0.5 * (start_dt + end_dt)

        idx = np.logical_and(utc_days < end_dt, utc_days > start_dt)
        utc_days[idx] = mean_dt

    groups = sorted(list(set(list(utc_days))))
    dates = np.array(
        [np.datetime64('01-01-01') + np.timedelta64(int(np.floor(group - 365.2422)), 'D') for group in groups])

    #prepare icesat arrays to pass to wrapper per stack, to avoid reading HDF5/NetCDF multiple times if doing parallel
    icesat_argsin = []
    for fn_stack in list_inters:
        ds = xr.open_dataset(fn_stack)
        tile_name = st.tilename_stack(ds)
        lat, lon = ot.SRTMGL1_naming_to_latlon(tile_name)
        pts_idx = np.logical_and.reduce((ice.lat > lat, ice.lat <= lat + 1, ice.lon > lon, ice.lon <= lon + 1))
        check = np.count_nonzero(pts_idx)
        print('Tile ' + tile_name + ': found ' + str(check) + ' ICESat points')
        if check > 0:
            _, utm = ot.latlon_to_UTM(lat, lon)
            ice.project('epsg:{}'.format(ot.epsg_from_utm(utm)))
            ice_coords = [tup for i, tup in enumerate(ice.xy) if pts_idx[i]]
            ice_elev = ice.elev[pts_idx]
            ice_date = ice.UTCTime[pts_idx]
            ice_latlon = np.array(list(zip(ice.lat[pts_idx],ice.lon[pts_idx])))

            icesat_argsin.append((fn_stack,np.copy(ice_coords),np.copy(ice_latlon),np.copy(ice_elev),np.copy(ice_date),np.copy(groups),np.copy(dates),read_filt,inc_mask,exc_mask))

    if nproc == 1:
        list_h, list_dh, list_zsc, list_dt, list_pos, list_slp, list_time, list_lat, list_lon, list_dh_ref, list_curv = ([] for i in range(10))
        for i in range(len(icesat_argsin)):
            tmp_h, tmp_dh, tmp_zsc, tmp_dt, tmp_pos, tmp_slp, tmp_time, tmp_lon, tmp_lat, tmp_dh_ref, tmp_curv = icesat_comp_wrapper(icesat_argsin[i])
            list_h.append(tmp_h)
            list_dh.append(tmp_dh)
            list_zsc.append(tmp_zsc)
            list_dt.append(tmp_dt)
            list_pos.append(tmp_pos)
            list_slp.append(tmp_slp)
            list_time.append(tmp_time)
            list_lat.append(tmp_lat)
            list_lon.append(tmp_lon)
            list_dh_ref.append(tmp_dh_ref)
            list_curv.append(tmp_curv)

        h = np.concatenate(list_h)
        dh = np.concatenate(list_dh)
        zsc = np.concatenate(list_zsc)
        dt_out = np.concatenate(list_dt)
        pos = np.concatenate(list_pos)
        slp = np.concatenate(list_slp)
        t = np.concatenate(list_time)
        lat = np.concatenate(list_lat)
        lon = np.concatenate(list_lon)
        dh_ref = np.concatenate(list_dh_ref)
        curv = np.concatenate(list_curv)
    else:
        nproc=min(len(icesat_argsin),nproc)
        print('Using '+str(nproc)+' processors...')
        pool = mp.Pool(nproc,maxtasksperchild=1)
        outputs = pool.map(icesat_comp_wrapper,icesat_argsin)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        h = np.concatenate(zip_out[0])
        dh = np.concatenate(zip_out[1])
        zsc = np.concatenate(zip_out[2])
        dt_out = np.concatenate(zip_out[3])
        pos = np.concatenate(zip_out[4])
        slp = np.concatenate(zip_out[5])
        t = np.concatenate(zip_out[6])
        lon = np.concatenate(zip_out[7])
        lat = np.concatenate(zip_out[8])
        dh_ref = np.concatenate(zip_out[9])
        curv = np.concatenate(zip_out[10])

    return h, dh, zsc, dt_out, pos, slp, t, lon, lat, dh_ref, curv

def shift_icesat_stack(fn_ref,fn_icesat,fn_shp):

    _ , _ , shift_params, stats = dem_coregistration(fn_icesat,fn_ref,glaciermask=fn_shp,pts=True,inmem=True)

    return shift_params, stats


def combine_postproc_stacks_tvol(list_fn_stack, fn_shp, feat_id='RGIId', tlim=None, write_combined=True, outdir='.'):
    # get all rgiid intersecting stacks and the list of intersecting stacks
    all_rgiids, list_list_stacks = inters_feat_shp_stacks(fn_shp, list_fn_stack, feat_id)

    # sort by rgiid group with same intersecting stacks
    list_tuples = list(zip(all_rgiids, list_list_stacks))
    grouped = [(k, list(list(zip(*g))[0])) for k, g in groupby(list_tuples, itemgetter(1))]

    # loop through similar combination of stacks (that way, only have to combine/open them once)
    for i in range(len(grouped)):

        list_fn_stack_pack = grouped[i][0]
        rgiid_pack = grouped[i][1]

        list_ds = st.open_datasets(list_fn_stack_pack)
        if len(list_ds) > 1:
            ds = st.combine_stacks(list_ds)
        else:
            ds = list_ds[0]

        if write_combined:
            list_tile = [os.path.splitext(os.path.basename(fn))[0].split('_')[0] for fn in list_fn_stack_pack]
            out_nc = os.path.join(outdir, 'combined_stacks', '_'.join(list_tile))
            ds.to_netcdf(out_nc)

        df_tot = pd.DataFrame()
        # loop through rggiids
        for rgiid in rgiid_pack:
            ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)
            layer_name = os.path.splitext(os.path.basename(fn_shp))[0]
            geoimg = st.make_geoimg(ds, 0)
            mask = ot.geoimg_mask_on_feat_shp_ds(ds_shp, geoimg, layer_name=layer_name, feat_id=feat_id, feat_val=rgiid)

            dc, submask, _ = sel_dc(ds, tlim, mask)
            df = int_dc(dc, submask)

            df_tot.append(df)


def get_dt_closest_valid(ds_filt,dates):

    # we read the boolean data cube indicating positions where original data was used

    # first, remove duplicate dates by merging boolean arrays for same dates
    t_vals = list(ds_filt.time.values)
    dates_rm_dupli = list(set(t_vals))
    ind_firstdate = []
    for i, date in enumerate(dates_rm_dupli):
        ind_firstdate.append(t_vals.index(date))
    ind_firstdate = sorted(ind_firstdate)
    ds_filt2 = ds_filt.isel(time=np.array(ind_firstdate))
    for i in range(len(dates_rm_dupli)):
        t_ind = (t_vals == dates_rm_dupli[i])
        if len(t_ind) > 1:
            ds_filt2.z.values[i, :] = np.any(~ds_filt.z[t_vals == dates_rm_dupli[i], :].values.astype(bool), axis=0)

    # getting time data as days since 2000
    y0 = np.datetime64('2000-01-01')
    ftime = ds_filt2.time.values
    ftime_delta = np.array([t - y0 for t in ftime])
    days = [td.astype('timedelta64[D]').astype(int) for td in ftime_delta]

    # reindex to get closest not-NaN time value of the date vector
    filt_arr = np.array(ds_filt2.z.values, dtype='float32')
    filt_arr[filt_arr == 0] = np.nan
    days = np.array(days)
    filt_arr = filt_arr * days[:, None, None]
    at_least_2=np.count_nonzero(~np.isnan(filt_arr),axis=0)>2
    filt_tmp = filt_arr[:,at_least_2]
    out_arr = np.copy(filt_tmp)
    for i in range(np.shape(filt_tmp)[1]):
        ind = ~np.isnan(filt_tmp[:,i])
        fn = interp1d(days[ind],filt_tmp[:,i][ind],kind='nearest',fill_value='extrapolate',assume_sorted=True)
        out_arr[:,i] = fn(days)
    filt_arr[:,at_least_2] = out_arr
    ds_filt2.z.values = filt_arr
    ds_filt_sub = ds_filt2.reindex(time=dates, method='nearest')

    for i in range(len(dates)):
        date = dates[i]
        day_diff = (date - y0).astype('timedelta64[D]').astype(int)
        ds_filt_sub.z.values[i,:] = ds_filt_sub.z.values[i, :] - np.ones(ds_filt_sub.z.shape[1:3]) * day_diff

    return ds_filt_sub

def sel_int_hypsocheat(argsin):

    # integrate volume without having to know the exact spatial disposition: works for hypsometric (needs only reference elevation)
    # a LOT faster as we don't need to combine (reproject + merge) stacks
    list_fn_stack_pack, shp_params, tlim, i = argsin

    print('Integrating volume for outline group: '+str(i+1))

    list_ds = st.open_datasets(list_fn_stack_pack,load=True)

    dates = list_ds[0].time.values

    list_ds_filt = []
    for fn_stack in list_fn_stack_pack:
        fn_filt = os.path.join(os.path.dirname(fn_stack), os.path.basename(fn_stack).split('_')[0] + '_filtered.nc')
        ds_filt = xr.open_dataset(fn_filt)
        ds_filt.load()
        list_ds_filt.append(ds_filt)

    fn_shp, feat_id, list_feat_val = shp_params
    ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)
    layer_name = os.path.splitext(os.path.basename(fn_shp))[0]

    layer = ds_shp.GetLayer()
    list_lat, list_lon = ([] for i in range(2))
    for feat_val in list_feat_val:
        for feature in layer:
            if feat_val == feature.GetField(feat_id):
                geom = feature.GetGeometryRef()
                centroid_lon, centroid_lat, _ = geom.Centroid().GetPoint()
                list_lon.append(centroid_lon)
                list_lat.append(centroid_lat)
        layer.ResetReading()

    df_tot, df_hyp_tot, df_int_tot = (pd.DataFrame() for i in range(3))
    for feat_val in list_feat_val:
        dh, err, ref, dt, count_area = ([] for i in range(5))
        print('Working on feature ID: '+feat_val)
        for i, ds in enumerate(list_ds):

            ds_filt = list_ds_filt[i]

            #get raster equivalent of stack

            geoimg = st.make_geoimg(ds, 0)
            #get mask of latlon tiling
            tile_name = st.tilename_stack(ds)
            mask_tile = ot.latlontile_nodatamask(geoimg,tile_name)
            mask_feat = ot.geoimg_mask_on_feat_shp_ds(ds_shp, geoimg, layer_name=layer_name, feat_id=feat_id, feat_val=feat_val)

            mask = np.logical_and(mask_tile,mask_feat)

            if np.count_nonzero(mask) >0:
                dc, submask, _ = sel_dc(ds,tlim,mask)
                dc_dt, _, _ = sel_dc(ds_filt,tlim,mask)

                dc_dt_sub = get_dt_closest_valid(dc_dt,dates)

                tmp_dh = dc.z.values[:,submask] - dc.z.values[0,submask]
                tmp_err = np.sqrt(dc.z_ci.values[:,submask]**2 + dc.z_ci.values[0,submask]**2)
                tmp_ref = dc.z.values[0,submask]
                tmp_dt = dc_dt_sub.z.values[:,submask]
                count = np.count_nonzero(mask)

                dh.append(tmp_dh)
                err.append(tmp_err)
                ref.append(tmp_ref)
                dt.append(tmp_dt)
                count_area.append(count)
            else:
                continue

        if len(dh)>0 and np.count_nonzero(~np.isnan(np.concatenate(dh,axis=1)))>0:
            dh = np.concatenate(dh,axis=1)
            err = np.concatenate(err,axis=1)
            ref = np.concatenate(ref)
            dt = np.concatenate(dt,axis=1)

            ds = list_ds[0]
            x = ds.x.shape[0]
            dx = np.round((ds.x.max().values - ds.x.min().values) / float(x))

            df, df_hyp, df_int = ot.hypso_dc(dh,err,ref,dt,dates,np.ones(np.shape(ref),dtype=bool),dx)

        elif len(dh)>0:
            dx = np.round((ds.x.max().values - ds.x.min().values) / float(x))
            area = np.sum(np.array(count_area))*dx**2
            df = pd.DataFrame()
            df = df.assign(hypso=[np.nan], time=[np.nan], dh=[np.nan], err_dh=[np.nan])
            df_hyp = pd.DataFrame()
            df_hyp = df_hyp.assign(hypso=[np.nan], area_meas=[np.nan], area_tot=[area], nmad=[np.nan])
            df_int = pd.DataFrame()
            df_int = df_int.assign(time=dates, dh=[np.nan]*len(dates), err_dh=[np.nan]*len(dates), area=[area]*len(dates))

        df['rgiid'] = feat_val
        df_hyp['rgiid'] = feat_val
        df_int['rgiid'] = feat_val

        df_int['lon'] = list_lon[list_feat_val.index(feat_val)]
        df_int['lat'] = list_lat[list_feat_val.index(feat_val)]

        df_tot = df_tot.append(df)
        df_hyp_tot = df_hyp_tot.append(df_hyp)
        df_int_tot = df_int_tot.append(df_int)

    return df_tot, df_hyp_tot, df_int_tot

def hypsocheat_postproc_stacks_tvol(list_fn_stack, fn_shp, feat_id='RGIId', tlim=None,nproc=64, outfile='int_dh.csv'):

    # get all rgiid intersecting stacks and the list of intersecting stacks
    start = time.time()

    all_rgiids, list_list_stacks = inters_feat_shp_stacks(fn_shp, list_fn_stack, feat_id)

    # sort by rgiid group with same intersecting stacks
    list_tuples = list(zip(all_rgiids, list_list_stacks))
    grouped = [(k, list(list(zip(*g))[0])) for k, g in groupby(list_tuples, itemgetter(1))]

    print('Found '+str(len(all_rgiids))+' outlines intersecting stacks')
    print('Grouped outlines in '+str(len(grouped))+' packs')

    print('Elapsed: '+str(time.time()-start))
    # loop through similar combination of stacks (that way, only have to combine/open them once)
    df_final, df_hyp_final, df_int_final = (pd.DataFrame() for i in range(3))
    if nproc == 1:

        for i in range(len(grouped)):
            list_fn_stack_pack = grouped[i][0]
            rgiid_pack = grouped[i][1]

            shp_params = (fn_shp,feat_id,rgiid_pack)
            df, df_hyp, df_int = sel_int_hypsocheat((list_fn_stack_pack,shp_params,tlim,i))
            df_final = df_final.append(df)
            df_hyp_final = df_hyp_final.append(df_hyp)
            df_int_final = df_int_final.append(df_int)
    else:
        argsin = [(grouped[i][0], (fn_shp, feat_id, grouped[i][1]), tlim, i) for i in range(len(grouped))]
        pool = mp.Pool(nproc,maxtasksperchild=1)
        outputs = pool.map(sel_int_hypsocheat,argsin,chunksize=1)
        pool.close()
        pool.join()

        print('Finished processing, elapsed: ' + str(time.time() - start))

        zips = list(zip(*outputs))

        dfs = zips[0]
        dfs_hyp = zips[1]
        dfs_int = zips[2]

        print('Finished zipping, elapsed: ' + str(time.time() - start))

        df_final = pd.concat(dfs)
        df_hyp_final = pd.concat(dfs_hyp)
        df_int_final = pd.concat(dfs_int)

        print('Finished putting in dataframes, elapsed: ' + str(time.time() - start))

    fn_csv = os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(outfile))[0]+'_all.csv')
    fn_hyp_csv =  os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(outfile))[0]+'_hyp.csv')
    fn_int_csv =  os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(outfile))[0]+'_int.csv')
    df_final.to_csv(fn_csv,index=False)
    df_hyp_final.to_csv(fn_hyp_csv,index=False)
    df_int_final.to_csv(fn_int_csv,index=False)

def aggregate_df_extents():
    pass

def aggregate_df_shp():
    pass

def aggregate_df_time(df,df_hyp):

    df = pd.read_csv('/home/atom/ongoing/int_dh_all.csv')
    df_hyp = pd.read_csv('/home/atom/ongoing/int_dh_hyp.csv')

    df['vol'] = df.merge(df_hyp).assign(vol=lambda df: df.dh * df.area_meas).vol



    df_tot = df.groupby('time',as_index=False)['vol'].sum()

