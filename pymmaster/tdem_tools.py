"""
pymmaster.tdem_tools provides tools to post-process DEM stacks (volume integration, etc...)
"""
from __future__ import print_function
import xarray as xr
import os
import sys
import numpy as np
from itertools import groupby
from operator import itemgetter
import multiprocessing as mp
import gdal
import osr
import ogr
import time
from datetime import datetime as dt
import pandas as pd
import pymmaster.stack_tools as st
import pymmaster.other_tools as ot
import pymmaster.fit_tools as ft
from pybob.coreg_tools import get_slope, create_stable_mask
from pybob.GeoImg import GeoImg
from pybob.ICESat import ICESat
from glob import glob


def inters_feat_shp_stacks(fn_shp, list_fn_stack, feat_field_name):
    # get intersecting rgiid for each stack extent
    list_list_rgiid = []
    for fn_stack in list_fn_stack:
        ds = xr.open_dataset(fn_stack)
        extent, proj = st.extent_stack(ds)
        list_rgiid = ot.list_shp_field_inters_extent(fn_shp, feat_field_name, extent, proj)
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

    return dc, submask

def int_dc(dc, mask, **kwargs):
    # integrate data cube over masked area
    dh = dc.variables['z'].values - dc.variables['z'].values[0]
    err = np.sqrt(dc.variables['z_ci'].values ** 2 + dc.variables['z_ci'].values[0] ** 2)
    ref_elev = dc.variables['z'].values[0]
    slope = get_slope(st.make_geoimg(dc, 0)).img
    slope[np.logical_or(~np.isfinite(slope), slope > 70)] = np.nan

    t, y, x = np.shape(dh)
    dx = np.round((dc.x.max().values - dc.x.min().values) / float(x))

    # df , _ = ot.hypso_dc()

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

        inters = poly_st.Intersection(poly)

        if not inters.IsEmpty():
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

def icesat_comp_wrapper(argsin):

    fn_stack,ice_coords,ice_elev,ice_date,groups,dates,read_filt,fn_shp = argsin

    full_dh = full_z_score = full_dt = full_pos = full_slp = np.array([])
    ds = xr.open_dataset(fn_stack)
    tile_name = st.tilename_stack(ds)

    tmp_h = st.make_geoimg(ds)
    tmp_ci = st.make_geoimg(ds)
    tmp_slope = st.make_geoimg(ds)
    tmp_slope.slope = ds.slope.values

    ds_sub = ds.interp(time=dates)

    if fn_shp is not None:

        print('Tile '+tile_name+': deriving terrain mask...')
        #rasterizing mask and changing to float to use geoimg.raster_points2
        mask = ft.get_stack_mask(fn_shp,ds)
        tmp_mask = st.make_geoimg(ds)
        tmp_mask.img = np.array(mask,dtype='float32')

    if read_filt:
        print('Tile '+tile_name+': getting original data dates from boolean array...')

        #we read the boolean data cube indicating positions where original data was used
        tmp_dt = st.make_geoimg(ds)
        fn_filt = os.path.join(os.path.dirname(fn_stack), os.path.basename(fn_stack).split('_')[0] + '_filtered.nc')
        ds_filt = xr.open_dataset(fn_filt)

        #first, remove duplicate dates by merging boolean arrays for same dates
        t_vals = list(ds_filt.time.values)
        dates_rm_dupli = list(set(t_vals))
        ind_firstdate = []
        for i, date in enumerate(dates_rm_dupli):
            ind_firstdate.append(t_vals.index(date))
        ind_firstdate = sorted(ind_firstdate)
        ds_filt2 = ds_filt.isel(time=np.array(ind_firstdate))
        for i in range(len(dates_rm_dupli)):
            ds_filt2.z.values[i,:] = np.any(ds_filt.z[t_vals==dates_rm_dupli[i],:].values,axis=0)

        #getting time data as days since 2000
        y0 = np.datetime64('2000-01-01')
        ftime = ds_filt2.time.values
        ftime_delta = np.array([t - np.datetime64('2000-01-01') for t in ftime])
        days = [td.astype('timedelta64[D]').astype(int) for td in ftime_delta]

        #reindex to get closest not-NaN time value for each of the ICESat laser campaign date
        filt_arr = np.array(ds_filt2.z.values,dtype='float32')
        filt_arr[filt_arr == 0] = np.nan
        days = np.array(days)
        filt_arr = filt_arr * days[:,None,None]
        ds_filt2.z.values = filt_arr
        ds_filt_sub = ds_filt2.reindex(time=dates,method='nearest')

    for i, group in enumerate(groups):
        #keep only a campaign group
        pts_idx_dt = np.array(ice_date == group)
        date = dates[i]

        print('Tile '+tile_name+': calculating differences for the ' + str(
            np.count_nonzero(pts_idx_dt)) + ' points of campaign:' + str(date))

        subsamp_ice = [tup for i, tup in enumerate(ice_coords) if pts_idx_dt[i]]

        tmp_h.img = ds_sub.z[i, :].values
        comp_pts_h = tmp_h.raster_points(subsamp_ice, nsize=5, mode='linear')

        tmp_ci.img = ds_sub.z_ci[i, :].values
        comp_pts_ci = tmp_ci.raster_points(subsamp_ice, nsize=5, mode='linear')

        comp_pts_slope = tmp_slope.raster_points(subsamp_ice, nsize=5, mode='linear')

        print(np.shape(comp_pts_h))
        print(np.shape(comp_pts_ci))
        print(np.shape(pts_idx_dt))
        print(np.shape(ice_elev[pts_idx_dt]))
        dh = ice_elev[pts_idx_dt] - comp_pts_h
        print(np.shape(dh))
        good_vals = np.isfinite(dh)
        dh = dh[good_vals]
        print(np.shape(dh))
        print(np.shape(good_vals))
        z_score = dh / comp_pts_ci[good_vals]
        slp = comp_pts_slope[good_vals]

        if read_filt:
            day_diff = (date - y0).astype('timedelta64[D]').astype(int)
            tmp_dt.img = ds_filt_sub.z[i, :].values - np.ones(np.shape(tmp_dt.img)) * day_diff
            comp_pts_dt = tmp_dt.raster_points(subsamp_ice, nsize=5, mode='mean')
            dt_out = comp_pts_dt[good_vals]
        else:
            dt_out = np.zeros(len(dh)) * np.nan

        if fn_shp is not None:
            comp_pts_mask = tmp_mask.raster_points(subsamp_ice, nsize=5, mode='nearest')
            pos = comp_pts_mask.astype(dtype=bool)
        else:
            pos = np.ones(len(dh),dtype=bool)

        full_dh = np.concatenate([full_dh, dh])
        full_z_score = np.concatenate([full_z_score, z_score])
        full_dt = np.concatenate([full_dt,dt_out])
        full_pos = np.concatenate([full_pos,pos])
        full_slp = np.concatenate([full_slp,slp])

    return full_dh, full_z_score, full_dt, full_pos, full_slp


def comp_stacks_icesat(list_fn_stack,fn_icesat,fn_shp=None,nproc=1,read_filt=False):

    ice = ICESat(fn_icesat)
    ice.clean(el_limit=-200)

    #get intersecting tiles
    bounds = ice.get_bounds()
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    proj_wkt = proj.ExportToWkt()
    list_inters = get_inters_stack_dem(list_fn_stack,bounds,proj_wkt)

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
        start_dt = datetime_to_yearfraction(laser_op_icesat[i][0]) * 365.2422 - 3
        end_dt = datetime_to_yearfraction(laser_op_icesat[i][1]) * 365.2422 + 3

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

            icesat_argsin.append((fn_stack,np.copy(ice_coords),np.copy(ice_elev),np.copy(ice_date),groups,dates,read_filt,fn_shp))

    if nproc == 1:
        list_dh = list_zsc = list_dt = list_pos = list_slp = []
        for i in range(len(icesat_argsin)):
            tmp_dh, tmp_zsc, tmp_dt, tmp_pos, tmp_slp = icesat_comp_wrapper(icesat_argsin[i])
            list_dh.append(tmp_dh)
            list_zsc.append(tmp_zsc)
            list_dt.append(tmp_dt)
            list_pos.append(tmp_pos)
            list_slp.append(tmp_slp)

        dh = np.concatenate(list_dh)
        zsc = np.concatenate(list_zsc)
        dt_out = np.concatenate(list_dt)
        pos = np.concatenate(list_pos)
        slp = np.concatenate(list_slp)
    else:
        nproc=min(len(list_inters),nproc)
        print('Using '+str(nproc)+' processors...')
        pool = mp.Pool(nproc,maxtasksperchild=1)
        outputs = pool.map(icesat_comp_wrapper,icesat_argsin)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        dh = np.concatenate(zip_out[0])
        zsc = np.concatenate(zip_out[1])
        dt_out = np.concatenate(zip_out[2])
        pos = np.concatenate(zip_out[3])
        slp = np.concatenate(zip_out[4])

    return dh, zsc, dt_out, pos, slp


def postproc_stacks_tvol(list_fn_stack, fn_shp, feat_id='RGIId', tlim=None, write_combined=True, outdir='.'):
    # get all rgiid intersecting stacks and the list of intersecting stacks
    all_rgiids, list_list_stacks = inters_feat_shp_stacks(fn_shp, list_fn_stack, feat_id)

    # sort by rgiid group with same intersecting stacks
    list_tuples = list(zip(all_rgiids, list_list_stacks))
    grouped = [(k, list(list(zip(*g))[0])) for k, g in groupby(list_tuples, itemgetter(1))]

    # loop through similar combination of stacks (that way, only have to combine them once)
    for i in range(len(grouped)):

        list_fn_stack_pack = grouped[0]
        rgiid_pack = grouped[1]

        list_ds = st.open_datasets(list_fn_stack_pack)
        if len(list_ds) > 1:
            ds = st.combine_stacks(list_ds)
        else:
            ds = list_ds[0]

        if write_combined:
            list_tile = [os.path.splitext(os.path.basename(fn))[0].split('_')[0] for fn in list_fn_stack_pack]
            out_nc = os.path.join(outdir, 'combined_stacks', '_'.join(list_tile))
            ds.to_netcdf(out_nc)

        # loop through rggiids
        for rgiid in rgiid_pack:
            ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)
            layer_name = os.path.splitext(os.path.basename(fn_shp))[0]
            geoimg = st.make_geoimg(ds, 0)
            mask = ot.geoimg_mask_on_feat_shp_ds(ds_shp, geoimg, layer_name=layer_name, feat_id=feat_id, feat_val=rgiid)

            dc, submask = sel_dc(ds, tlim, mask)
            int_dc(dc, submask)
