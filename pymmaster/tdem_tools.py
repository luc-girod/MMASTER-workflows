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
import gdal
import pymmaster.stack_tools as st
import pymmaster.other_tools as ot
from pybob.coreg_tools import get_slope


def inters_shp_stacks(fn_shp, list_fn_stack, field_name):
    # get intersecting rgiid for each stack extent
    list_list_rgiid = []
    for fn_stack in list_fn_stack:
        ds = xr.open_dataset(fn_stack)
        extent, proj = st.extent_stack(ds)
        list_rgiid = ot.list_shp_field_inters_extent(fn_shp, field_name, extent, proj)
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

    # for i in range(t):
    #     df , _ = vol_hypso_linear(dh[i,:,:],ref_elev,mask,dx,slope)


def postproc_stacks_tvol(list_fn_stack, fn_shp, feat_id='RGIId', tlim=None, write_combined=True, outdir='.'):
    # get all rgiid intersecting stacks and the list of intersecting stacks
    all_rgiids, list_list_stacks = inters_shp_stacks(fn_shp, list_fn_stack, feat_id)

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
