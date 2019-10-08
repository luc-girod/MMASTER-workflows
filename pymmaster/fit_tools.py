from __future__ import print_function
import os
import sys
import time
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
import numpy as np
import gdal
import pandas as pd
import xarray as xr
import matplotlib.pylab as plt
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import chain
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import filters
from skimage.morphology import disk
from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, ExpSineSquared as ESS, PairwiseKernel
from numba import jit
from llc import jit_filter_function
from pybob.GeoImg import GeoImg
from pybob.coreg_tools import get_slope
from pybob.image_tools import create_mask_from_shapefile
from pybob.plot_tools import set_pretty_fonts
# from pymmaster.stack_tools import create_crs_variable, create_nc
import pymmaster.stack_tools as st
import pymmaster.tdem_tools as tt
from pybob.ddem_tools import nmad
from warnings import filterwarnings

filterwarnings('ignore')

def make_dh_animation(ds, month_a_year=None, figsize=(8,10), t0=None, t1=None, dh_max=20, var='z', cmap='RdYlBu', xlbl='easting (km)',
                      ylbl='northing (km)'):
    set_pretty_fonts()
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    ds_sub = ds.sel(time=slice(t0, t1))

    if month_a_year is not None:
        t_vals = ds_sub.time.values
        y0 = t_vals[0].astype('datetime64[D]').astype(object).year
        y1 = t_vals[-1].astype('datetime64[D]').astype(object).year

        t_vec=[]
        for y in np.arange(y0,y1,1):
            t=np.datetime64(str(y)+'-'+str(month_a_year).zfill(2)+'-01')
            td=np.array([(t-t_vals[i].astype('datetime64[D]')).astype(int) for i in range(len(t_vals))])

            closer_dat = t_vals[(np.abs(td) == np.min(np.abs(td)))][0]
            t_vec.append(closer_dat)


        ds_sub = ds.sel(time=t_vec)
        # mid = int(np.floor(len(ds_sub.time.values)/2))

    if var == 'z':
        dh_ = ds_sub.variables[var].values - ds_sub.variables[var].values[0]
    elif var == 'z_ci':
        dh_ = ds_sub.variables[var].values

    times = np.array([np.datetime_as_string(t.astype('datetime64[D]')) for t in ds_sub.time.values])
    nice_ext = np.array([ds.x.values.min(), ds.x.values.max(), ds.y.values.min(), ds.y.values.max()]) / 1000
    ims = []

    im = ax.imshow(dh_[0], extent=nice_ext, vmin=-dh_max, vmax=dh_max, cmap=cmap)
    ann = ax.annotate(times[0], xy=(0.05, 0.05), xycoords='axes fraction', fontsize=20,
                      fontweight='bold', color='black', family='monospace')
    ims.append([im, ann])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, extend='both')

    cax.set_ylabel('elevation change (m)')
    ax.set_ylabel(ylbl)
    ax.set_xlabel(xlbl)

    for i in range(len(times[1:])):
        im = ax.imshow(dh_[i+1], vmin=-dh_max, vmax=dh_max, cmap=cmap, extent=nice_ext)
        ann = ax.annotate(times[i+1], xy=(0.05, 0.05), xycoords='axes fraction', fontsize=20,
                          fontweight='bold', color='black', family='monospace')
        ims.append([im, ann])

    return fig, ims


def write_animation(fig, ims, outfilename='output.gif', ani_writer='imagemagick', **kwargs):
    ani = animation.ArtistAnimation(fig, ims, **kwargs)
    ani.save(outfilename, writer=ani_writer)

def get_full_dh(ds,t0,t1,outname):

    tmp_img = st.make_geoimg(ds)

    dc = ds.sel(time=slice(t0,t1))

    dh = dc.variables['z'].values[-1] - dc.variables['z'].values[0]
    err = np.sqrt(dc.variables['z_ci'].values[-1] ** 2 + dc.variables['z_ci'].values[0] ** 2)
    # slope = ds.slope.values

    tmp_img.img = dh
    tmp_img.write(os.path.join(os.path.dirname(outname),os.path.basename(outname)+'_dh.tif'))
    tmp_img.img = err
    tmp_img.write(os.path.join(os.path.dirname(outname),os.path.basename(outname)+'_err.tif'))
    # tmp_img.img = slope
    # tmp_img.write(os.path.join(os.path.dirname(outname),os.path.basename(outname)+'_slope.tif'))


def get_stack_mask(maskshp, ds):
    npix_y, npix_x = ds['z'][0].shape
    dx = np.round((ds.x.max().values - ds.x.min().values) / float(npix_x))
    dy = np.round((ds.y.min().values - ds.y.max().values) / float(npix_y))

    newgt = (ds.x.min().values - 0, dx, 0, ds.y.max().values - 0, 0, dy)

    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', npix_x, npix_y, 1, gdal.GDT_Float32)

    sp = dst.SetProjection(ds.crs.spatial_ref)
    sg = dst.SetGeoTransform(newgt)

    wa = dst.GetRasterBand(1).WriteArray(ds['z'][0].values)
    md = dst.SetMetadata({'Area_or_point': 'Point'})
    del wa, sg, sp, md

    img = GeoImg(dst)
    mask = create_mask_from_shapefile(img, maskshp)
    return mask

def vgm_1d(argsin):

    t_vals, detrend_elev, lag_cutoff, tstep = argsin
    # 1D variogram: inspired by http://connor-johnson.com/2014/03/20/simple-kriging-in-python/

    def SVh(P, h, bw):
        # empirical variogram for a single lag
        pd = np.abs(np.subtract.outer(P[:,0], P[:,0]))
        N = pd.shape[0]
        Z = list()
        for i in range(N):
            for j in range(i + 1, N):
                if (pd[i, j] >= h - bw) and (pd[i, j] <= h + bw):
                    Z.append((P[i, 1] - P[j, 1]) ** 2.0)
        if len(Z)>0:
            return np.nansum(Z) / (2.0 * len(Z)), len(Z)
        else:
            return np.nan, 0

    def SV(P, hs, bw):
        # empirical variogram for a collection of lags
        sv = list()
        p = list()
        for h in hs:
            ph, svh = SVh(P, h, bw)
            sv.append(svh)
            p.append(ph)
        sv = [[p[i], sv[i]] for i in range(len(hs))]
        return np.array(sv).T

    ind_valid = ~np.isnan(detrend_elev)
    sample = np.column_stack((t_vals[ind_valid], detrend_elev[ind_valid]))

    hs = np.arange(0, lag_cutoff, tstep)
    sv = SV(sample, hs, tstep)

    return sv

def wrapper_vgm1d(argsin):

    t_vals, stack_detrend_elev, k, lag_cutoff, tstep = argsin

    _ , nb_iter = np.shape(stack_detrend_elev)

    print('Pack of '+str(nb_iter)+' variograms number '+str(k))

    lags = np.arange(0,lag_cutoff,tstep)

    vdata = np.zeros((len(lags), nb_iter))
    pdata = np.zeros((len(lags), nb_iter))
    for i in np.arange(nb_iter):
        sv = vgm_1d((t_vals,stack_detrend_elev[:,i],lag_cutoff,tstep))
        vdata[:, i] = sv[0]
        pdata[:, i] = sv[1]

    return vdata, pdata

def draw_vgm_by_dh(fn_dh,bin_dh,fn_stack,shp_mask,outfile):

    dh = GeoImg(fn_dh).img

    df_all = pd.DataFrame()
    for i in range(len(bin_dh)-1):

        print('Bin of dh from '+str(bin_dh[i])+' to '+str(bin_dh[i+1]))

        add_mask = np.logical_and(np.abs(dh) >= bin_dh[i],np.abs(dh)<bin_dh[i+1])
        lags, vmean, vstd = estimate_vgm(fn_stack,shp_mask=shp_mask,rast_mask=add_mask,nproc=64,nsamp=10000)
        id = str(bin_dh[i]) + '-' + str(bin_dh[i+1])

        df = pd.DataFrame()
        df = df.assign(lags=lags, vmean=vmean, vstd=vstd)
        df = df.assign(id=[id]*len(df.index))
        df_all = df_all.append(df,ignore_index=True)

    df_all.to_csv(outfile)


def estimate_vgm(fn_stack,shp_mask=None,rast_mask=None,nsamp=10000,tstep=0.25,lag_cutoff=None,min_obs=8,nproc=1,pack_size=50):

    # estimate 1D variogram for multiple pixels: random sampling within mask

    # load filtered stack
    print('Loading stack '+fn_stack)
    ds = xr.open_dataset(fn_stack)

    # ds.load()
    ds_arr = ds.variables['z'].values

    # rasterize mask
    mask = np.ones(np.shape(ds_arr[0]),dtype=bool)

    if shp_mask is not None:
        mask = np.logical_and(get_stack_mask(shp_mask, ds),mask)

    if rast_mask is not None:
        mask = np.logical_and(rast_mask,mask)

    # count number of valid temporal observation for each pixel
    nb_arr = np.nansum(~np.isnan(ds_arr),axis=0)
    mask = np.logical_and(mask,nb_arr>=min_obs)

    print('Selecting a subsample of '+str(nsamp)+' points with at least '+str(min_obs)+ ' observations in time...')
    # sample a subset
    max_samp = np.count_nonzero(mask)
    index = np.where(mask)
    final_nsamp = min(max_samp,nsamp)
    subset = np.random.choice(max_samp,final_nsamp,replace=False)
    index_subset=(index[0][subset],index[1][subset])
    mask_subset = np.zeros(np.shape(mask),dtype=np.bool)
    mask_subset[index_subset] = True

    ds_samp = ds_arr[:, mask_subset]
    # ds_samp = ds.isel(y=index[0][subset],x=index[1][subset])['z'].values

    # read and convert time values
    t_vals = ds['time'].values

    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in t_vals])
    t_scale = (ftime_delta / total_delta) * (int(y1) - y0)

    if lag_cutoff is None:
        lag_cutoff = np.max(t_scale) - np.min(t_scale)

    lags = np.arange(0, lag_cutoff, tstep) + 0.5*tstep

    # get variance/lags and number of pairwise/lags for each pixel

    vdata = np.zeros((len(lags), final_nsamp))
    pdata = np.zeros((len(lags), final_nsamp))
    if nproc==1:
        print('Drawing variograms with 1 core...')
        for i in np.arange(final_nsamp):
            sv = vgm_1d((t_scale, ds_samp[:,i].flatten(),lag_cutoff,tstep))
            vdata[:,i]=sv[0]
            pdata[:,i]=sv[1]
    else:
        print('Drawing variograms with '+str(nproc)+ ' cores...')
        pool = mp.Pool(nproc, maxtasksperchild=1)
        argsin = [(t_scale,ds_samp[:,i:min(i+pack_size,final_nsamp)],k,lag_cutoff,tstep) for k ,i in enumerate(np.arange(0,final_nsamp,pack_size))]
        outputs = pool.map(wrapper_vgm1d, argsin, chunksize=1)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        for k, i in enumerate(np.arange(0,final_nsamp,pack_size)):
            vdata[:, i:min(i+pack_size,final_nsamp)]=zip_out[0][k]
            pdata[:, i:min(i+pack_size,final_nsamp)]=zip_out[1][k]

    ptot = np.nansum(pdata,axis=1)
    # mean variogram accounting for the number of pairwise comparison in each pixel
    vmean = np.nansum(vdata * pdata,axis=1) / ptot
    #'rough' std: between pixels, not accounting for the number of pairwise observation
    vstd = np.nanstd(vdata,axis=1)

    return lags, vmean, vstd

def plot_vgm(lags,vmean,vstd):

    fig, ax = plt.subplots(1)
    ax.plot(lags, vmean, lw=2, label='mean', color='blue')
    ax.fill_between(lags, vmean + vstd, vmean - vstd, facecolor='blue', alpha=0.5)
    ax.set_title('Variogram: ')
    ax.set_xlabel('Lag [year]')
    ax.set_ylabel('Semivariance [$\mu$ $\pm \sigma$]')
    ax.legend(loc='lower left')
    ax.grid()
    # plt.savefig(fn_fig, dpi=600)
    plt.show()

def parse_epsg(wkt):
    return int(''.join(filter(lambda x: x.isdigit(), wkt.split(',')[-1])))


def ols_matrix(X, Y, conf_interv=0.68, conf_slope=0.68):
    # perform independent OLS matricially for optimal processing time
    # inspired from: https://en.wikipedia.org/wiki/Simple_linear_regression#Normality_assumption
    # and https://fr.wikipedia.org/wiki/M%C3%A9thode_des_moindres_carr%C3%A9s
    x = X * 1.0
    y = Y * 1.0

    x[np.isnan(y)] = np.nan  # check for NaNs
    y[np.isnan(x)] = np.nan  # check for NaNs

    moy_X = np.nanmean(x, axis=0)
    moy_Y = np.nanmean(y, axis=0)

    mat_cross_product = (x - moy_X) * (y - moy_Y)
    sum_mat_cross_product = np.nansum(mat_cross_product, axis=0)

    mat_X_squared = (x - moy_X) ** 2
    sum_mat_X_squared = np.nansum(mat_X_squared, axis=0)

    beta1 = sum_mat_cross_product / sum_mat_X_squared
    beta0 = moy_Y - beta1 * moy_X

    # confidence interval
    alpha_interv = 1. - conf_interv
    alpha_slope = 1. - conf_slope

    Y_pred = beta1 * x + beta0
    n = np.sum(~np.isnan(x), axis=0)
    SSX = sum_mat_X_squared
    SXY = np.sqrt(np.nansum((y - Y_pred) ** 2, axis=0) / (n - 2))
    SE_slope = SXY / np.sqrt(SSX)
    hi = 1. / n + (x - moy_X) ** 2 / SSX

    # quantile of student's t distribution for p=1-alpha/2
    q_interv = stats.t.ppf(1. - alpha_interv / 2, n - 2)
    q_slope = stats.t.ppf(1. - alpha_slope / 2, n - 2)

    # upper and lower CI:
    dy = q_interv * SXY * np.sqrt(hi)
    Yl = Y_pred - dy
    Yu = Y_pred + dy

    # incert on slope
    incert_slope = q_slope * SE_slope

    return beta1, beta0, incert_slope, Yl, Yu


def wls_matrix(x, y, w, conf_interv=0.68, conf_slope=0.68):
    # perform independent WLS matricially for optimal processing time

    X = x * 1.0
    Y = y * 1.0
    W = w * 1.0

    Y[np.isnan(W) | np.isnan(X)] = np.nan  # check for NaNs
    X[np.isnan(W) | np.isnan(Y)] = np.nan  # check for NaNs
    W[np.isnan(Y) | np.isnan(X)] = np.nan  # check for NaNs

    sum_w = np.nansum(W, axis=0)
    moy_X_w = np.nansum(X * W, axis=0) / sum_w
    moy_Y_w = np.nansum(Y * W, axis=0) / sum_w

    mat_cross_product = W * (X - moy_X_w) * (Y - moy_Y_w)
    sum_mat_cross_product = np.nansum(mat_cross_product, axis=0)

    mat_X_squared = W * (X - moy_X_w) ** 2
    sum_mat_X_squared = np.nansum(mat_X_squared, axis=0)

    beta1 = sum_mat_cross_product / sum_mat_X_squared
    beta0 = moy_Y_w - beta1 * moy_X_w

    # confidence interval
    alpha_interv = 1. - conf_interv
    alpha_slope = 1. - conf_slope

    Y_pred = beta1 * X + beta0
    n = np.sum(~np.isnan(X), axis=0)
    SSX = sum_mat_X_squared
    SXY = np.sqrt(np.nansum(W * (Y - Y_pred) ** 2, axis=0) / (n - 2))
    SE_slope = SXY / np.sqrt(SSX)
    hi = 1. / n + W * (X - moy_X_w) ** 2 / SSX

    # quantile of student's t distribution for p=1-alpha/2
    q_interv = stats.t.ppf(1. - alpha_interv / 2, n - 2)
    q_slope = stats.t.ppf(1. - alpha_slope / 2, n - 2)

    # get the upper and lower CI:
    dy = q_interv * SXY * np.sqrt(hi)
    Yl = Y_pred - dy
    Yu = Y_pred + dy

    # calculate incert on slope
    incert_slope = q_slope * SE_slope

    return beta1, beta0, incert_slope, Yl, Yu


def interp_data(t, y, sig, interp_t):
    y_ = interp1d(t, y)
    s_ = interp1d(t, sig)
    return y_(interp_t), s_(interp_t)


def detrend(t_vals, data, ferr):
    n_out = 1
    while n_out > 0:
        reg = LinearRegression().fit(t_vals.reshape(-1, 1), data.reshape(-1, 1))

        trend = reg.predict(t_vals.reshape(-1, 1)).squeeze()
        std_nmad_rat = np.std(data - trend) / nmad(data - trend)
        if std_nmad_rat > 20:
            isin = np.abs(data - trend) < 4 * nmad(data - trend)
        else:
            isin = np.abs(data - trend) < 4 * np.std(data - trend)
        n_out = np.count_nonzero(~isin)

        data = data[isin]
        t_vals = t_vals[isin]
        ferr = ferr[isin]

    return reg

def gpr(data, t_vals, uncert, t_pred, opt=False, kernel=None, not_stable=True, detrend_ls=False, loop_detrend=False):

    # if only 0 or 1 elevation values in the pixel, no fitting
    if np.count_nonzero(np.isfinite(data)) < 2:
        return np.nan * np.zeros(t_pred.shape), np.nan * np.zeros(t_pred.shape), np.nan * np.zeros(data.shape)

    #converting time values
    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    ftime = t_vals[np.isfinite(data)]
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in ftime])
    time_vals = (ftime_delta / total_delta) * (int(y1) - y0)

    data_vals = data[np.isfinite(data)]
    err_vals = uncert[np.isfinite(data)]**2

    # by default, no optimizer: applying GPR with defined kernels
    if opt:
        optimizer = 'fmin_l_bfgs_b'
        n_restarts_optimizer = 9
    else:
        optimizer = None
        n_restarts_optimizer = 0

    # default kernels
    if kernel is None:
        # k1 = C(200) * RBF(10)  # other kernels to try to add here?
        # k2 = C(50) * RBF(1)
        # # k3 = C(10, (1e-3, 1e3)) * RQ(length_scale=30, length_scale_bounds=(30, 1e3))
        # kernel = k1 + k2

        # linear kernel + periodic kernel + local kernel
        k1 = PairwiseKernel(1, metric='linear')  # linear kernel
        k2 = C(30) * ESS(length_scale=1, periodicity=1)  # periodic kernel
        # k3 =  #local kernel
        k3 = C(30) * RBF(1)
        kernel = k1 + k2
        if not_stable:
            # k3 =  #non-linear kernel
            kernel += k3

    # initializing
    n_out = 1
    niter = 0
    tag_detr = 1

    num_finite = data_vals.size
    good_vals = np.isfinite(data_vals)
    max_z_score = [20,12,9,6,4]

    mu_x = np.mean(t_pred)

    # here we need to change the 0 for the x axis, in case we are using a linear kernel
    t_pred = t_pred - mu_x
    time_vals = time_vals - mu_x

    while n_out > 0 and num_finite >= 2:

        mu_y = np.nanmean(data_vals)

        if detrend_ls:
            # first, remove a linear trend
            if tag_detr != 0:
                try:
                    # try to remove a linear fit from the data before we fit, then add it back in when we're done.
                    reg = detrend(time_vals[good_vals], data_vals[good_vals], err_vals[good_vals])
                except:
                    return np.nan * np.zeros(t_pred.shape), np.nan * np.zeros(t_pred.shape), np.nan * np.zeros(data.shape)

                l_trend = reg.predict(time_vals.reshape(-1, 1)).squeeze()
                if not loop_detrend:
                    tag_detr = 0

            detr_data_vals = data_vals - l_trend
        else:
            # the mean has to be 0 to do gpr, even if we don't detrend
            detr_data_vals = data_vals - mu_y

        # if we remove a linear trend, normalize_y should be false...
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                                      alpha=err_vals[good_vals], normalize_y=False)
        gp.fit(time_vals[good_vals].reshape(-1, 1), detr_data_vals[good_vals].reshape(-1, 1))
        y_pred, sigma = gp.predict(t_pred.reshape(-1, 1), return_std=True)
        y_, s_ = interp_data(t_pred, y_pred.squeeze(), sigma.squeeze(), time_vals)
        z_score = np.abs(detr_data_vals - y_) / s_

        isin = np.logical_or(z_score < 4, ~np.isfinite(z_score))
        #we continue the loop if there is a least one value outside 4 stds
        n_out = np.count_nonzero(~isin)

        # good elevation values can also be outside 4stds because of bias in the first fits
        # thus, if needed, we remove outliers packet by packet, starting with the largest ones
        isout = np.logical_and(z_score > max_z_score[min(niter,len(max_z_score)-1)],np.isfinite(z_score))
        data_vals[isout] = np.nan

        good_vals = np.isfinite(data_vals)
        num_finite = np.count_nonzero(good_vals)
        niter += 1

    # if there is not enough data left...
    if num_finite < 2:
        y_pred = np.nan * np.zeros(t_pred.shape)
        sigma = np.nan * np.zeros(t_pred.shape)

    if detrend_ls:
        l_pred = reg.predict(t_pred.reshape(-1, 1)).squeeze()
        y_out = y_pred.squeeze() + l_pred
    else:
        y_out = y_pred.squeeze() + mu_y

    filt_data = data
    filt_data[np.isfinite(data)] = data_vals

    return y_out, sigma.squeeze(), np.isfinite(filt_data)


def ls(subarr, t_vals, uncert, weigh, filt_ls=False, conf_filt=0.99):

    T, Y, X = subarr.shape

    #converting time values
    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in t_vals])
    time_vals = y0 + (ftime_delta / total_delta) * (int(y1) - y0)

    z_mat = subarr.reshape(T, Y * X)
    t_mat = np.array([time_vals, ] * Y * X).T
    if weigh:
        w_mat = np.array([1. / uncert ** 2, ] * Y * X).T

    if filt_ls:
        if weigh:
            yl, yu = wls_matrix(t_mat, z_mat, w_mat, conf_interv=conf_filt)[3:5]
        else:
            yl, yu = ols_matrix(t_mat, z_mat, conf_interv=conf_filt)[3:5]

        z_mat[z_mat < yl] = np.nan
        z_mat[z_mat > yu] = np.nan

    if weigh:
        beta1, beta0, incert_slope = wls_matrix(t_mat, z_mat, w_mat)[0:3]
    else:
        beta1, beta0, incert_slope = ols_matrix(t_mat, z_mat)[0:3]

    date_min = np.nanmin(t_mat, axis=0)
    date_max = np.nanmax(t_mat, axis=0)
    nb_trend = (~np.isnan(z_mat)).sum(axis=0)

    filter_less_2_DEMs = nb_trend <= 2
    beta1[filter_less_2_DEMs] = np.nan
    incert_slope[filter_less_2_DEMs] = np.nan

    slope = np.reshape(beta1, (Y, X))
    interc = np.reshape(beta0, (Y, X))
    slope_sig = np.reshape(incert_slope, (Y, X))
    nb_dem = np.reshape(nb_trend, (Y, X))
    date_min = np.reshape(date_min, (Y, X))
    date_max = np.reshape(date_max, (Y, X))

    filt_subarr = np.isfinite(z_mat.reshape(T, Y, X))

    outarr = np.stack((slope, interc, slope_sig, nb_dem, date_min, date_max),axis=0)

    return outarr, filt_subarr

@jit
def gpr_wrapper(argsin):
    subarr, i, t_vals, med_slope, uncert, new_t, opt, kernel, uns_arr = argsin
    start = time.time()

    # get std, dependent on slope
    err_arr = np.ones(np.shape(subarr))
    err_arr = err_arr * uncert[:,None,None]
    err_arr = np.sqrt(err_arr ** 2 + (15 * np.tan(med_slope * np.pi / 180)) ** 2)
    err_arr[np.logical_or(~np.isfinite(err_arr), np.abs(err_arr) > 100)] = 100

    Y, X = subarr[0].shape
    outarr = np.nan * np.zeros((new_t.size * 2, Y, X))
    filt_subarr = np.zeros(np.shape(subarr),dtype=bool)
    # pixel by pixel
    for x in range(X):
        for y in range(Y):
            if uns_arr is not None:
                uns_tag = uns_arr[0,y,x]
            else:
                uns_tag = True
            uncert = err_arr[:,y,x]
            tmp_y, tmp_sig, tmp_filt = gpr(subarr[:, y, x], t_vals, uncert, new_t, opt=opt, not_stable=uns_tag, kernel=kernel)[0:3]
            out = np.concatenate((tmp_y, tmp_sig), axis=0)
            filt_subarr[: , y, x] = tmp_filt
            outarr[:, y, x] = out
    elaps = time.time() - start

    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr, filt_subarr

@jit
def ls_wrapper(argsin):
    subarr, i, t_vals, uncert, weigh, filt_ls, conf_filt = argsin
    start = time.time()
    # matrix
    outarr, filt_subarr = ls(subarr, t_vals, uncert, weigh, filt_ls=filt_ls, conf_filt=conf_filt)
    elaps = time.time() - start
    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr, filt_subarr


def splitter(img, nblocks):
    split1 = np.array_split(img, nblocks[0], axis=1)
    split2 = [np.array_split(im, nblocks[1], axis=2) for im in split1]
    olist = [np.copy(a) for a in list(chain.from_iterable(split2))]
    return olist


def stitcher(outputs, nblocks):
    stitched = []
    if np.array(nblocks).size == 1:
        nblocks = np.array([nblocks, nblocks])
    for i in range(nblocks[0]):
        stitched.append(outputs[i * nblocks[1]])
        for j in range(1, nblocks[1]):
            outind = j + i * nblocks[1]
            stitched[i] = np.concatenate((stitched[i], outputs[outind]), axis=2)
    return np.concatenate(tuple(stitched), axis=1)

def patchify(arr,nblocks,overlap):

    overlap = int(np.floor(overlap))
    patches = []
    nx, ny = np.shape(arr)
    nx_sub = nx // nblocks[0]
    ny_sub = ny // nblocks[1]
    split = [[nx_sub*i,min(nx_sub*(i+1),nx),ny_sub*j,min(ny_sub*(j+1),ny)] for i in range(nblocks[0]+1) for j in range(nblocks[1]+1)]
    over = [[max(0,l[0]-overlap),min(nx,l[1]+overlap),max(0,l[2]-overlap),min(l[3]+overlap,ny)] for l in split]
    inv = []
    for k in range(len(split)):

        x0, x1, y0, y1 = split[k]
        i0, i1, j0, j1 = over[k]
        patches.append(arr[i0:i1,j0:j1])
        inv.append([x0-i0,x1-i0,y0-j0,y1-j0])

    return patches, inv, split

def unpatchify(arr_shape, subarr, inv, split):

    out = np.zeros(arr_shape)

    for k, arr in enumerate(subarr):

        s = split[k]
        i = inv[k]
        out[s[0]:s[1],s[2]:s[3]] = arr[i[0]:i[1],i[2]:i[3]]

    return out

def cube_to_stack(ds, out_cube, y0, nice_fit_t, outfile, slope_arr=None, ci=True, clobber=False, filt_bool=False):

    fit_cube = out_cube[:len(nice_fit_t), :, :]

    img_shape=np.zeros(np.shape(fit_cube)[1:3])

    nco, to, xo, yo = st.create_nc(img_shape, outfile=outfile,
                                   clobber=clobber, t0=np.datetime64('{}-01-01'.format(y0)))
    st.create_crs_variable(parse_epsg(ds['crs'].spatial_ref), nco)

    x, y = ds['x'].values, ds['y'].values
    xo[:] = x
    yo[:] = y
    to[:] = nice_fit_t

    if not filt_bool:
        dt = 'f4'
        fill = -9999
    else:
        dt = 'i1'
        fill = 0
        fit_cube = np.array(fit_cube,dtype=bool)

    zo = nco.createVariable('z', dt, ('time', 'y', 'x'), fill_value=fill)
    zo.units = 'meters'
    zo.long_name = 'Fit elevation above WGS84 ellipsoid'
    zo.grid_mapping = 'crs'
    zo.coordinates = 'x y'
    zo.set_auto_mask(True)

    zo[:] = fit_cube

    if ci:
        sig_cube = out_cube[len(nice_fit_t):, :, :]
        fzo = nco.createVariable('z_ci', 'f4', ('time', 'y', 'x'), fill_value=-9999)
        fzo.units = 'meters'
        fzo.long_name = '68% confidence interval for elevation fit.'
        fzo.grid_mapping = 'crs'
        fzo.coordinates = 'x y'
        fzo.set_auto_mask(True)

        fzo[:] = sig_cube

    if slope_arr is not None:
        so = nco.createVariable('slope','f4', ('y','x'),fill_value=-9999)
        so.units = 'degrees'
        so.long_name = 'median slope used to condition elevation uncertainties'
        so.grid_mapping = 'crs'
        so.coordinates = 'x y'

        so[:] = slope_arr

    nco.close()


def arr_to_img(ds, out_arr, outfile):
    outfile_slope = os.path.join(os.path.dirname(outfile),
                               os.path.splitext(os.path.basename(outfile))[0] + '_dh.tif')
    outfile_interc = os.path.join(os.path.dirname(outfile),
                                 os.path.splitext(os.path.basename(outfile))[0] + '_interc.tif')
    outfile_sig = os.path.join(os.path.dirname(outfile),
                               os.path.splitext(os.path.basename(outfile))[0] + '_err.tif')
    outfile_nb = os.path.join(os.path.dirname(outfile),
                              os.path.splitext(os.path.basename(outfile))[0] + '_nb.tif')
    outfile_dmin = os.path.join(os.path.dirname(outfile),
                                os.path.splitext(os.path.basename(outfile))[0] + '_dmin.tif')
    outfile_dmax = os.path.join(os.path.dirname(outfile),
                                os.path.splitext(os.path.basename(outfile))[0] + '_dmax.tif')

    arr = st.make_geoimg(ds,band=0)
    arr.img = out_arr[0, :, :]
    arr.write(outfile_slope)
    arr.img = out_arr[1, :, :]
    arr.write(outfile_interc)
    arr.img = out_arr[2, :, :]
    arr.write(outfile_sig)
    arr.img = out_arr[3, :, :]
    arr.write(outfile_nb)
    arr.img = out_arr[4, :, :]
    arr.write(outfile_dmin)
    arr.img = out_arr[5, :, :]
    arr.write(outfile_dmax)


def time_filter_ref(ds, t_vals, ref_dem, ref_date, thresh=[-10,3], base_thresh=20):
    delta_t = (ref_date - t_vals).astype('timedelta64[D]').astype(float) / 365.24
    dh = ref_dem.img - ds
    dt_arr = np.ones(dh.shape)
    for i, d in enumerate(delta_t):
        dt_arr[i] = dt_arr[i] * d
    if np.array(thresh).size == 1:
        ds[np.abs(dh) > base_thresh + np.abs(dt_arr)*thresh] = np.nan
    else:
        d_data = dh / dt_arr
        ds[np.logical_and(d_data < - base_thresh / np.abs(dt_arr) + thresh[0], d_data > base_thresh / np.abs(dt_arr) + thresh[1])] = np.nan
    return ds

@jit_filter_function
def nanmax(a):
    return np.nanmax(a)

@jit_filter_function
def nanmin(a):
    return np.nanmin(a)

def wrapper_slope(argsin):

    arr, in_met = argsin

    #create input image
    gt, proj, npix_x, npix_y = in_met
    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', npix_x, npix_y, 1, gdal.GDT_Float32)
    sp = dst.SetProjection(proj)
    sg = dst.SetGeoTransform(gt)
    out_arr = np.copy(arr)
    out_arr[np.isnan(out_arr)] = -9999
    wa = dst.GetRasterBand(1).WriteArray(out_arr)
    md = dst.SetMetadata({'Area_or_point': 'Point'})
    nd = dst.GetRasterBand(1).SetNoDataValue(-9999)
    del sp, sg, wa, md, nd
    tmp_z = GeoImg(dst)

    slope = get_slope(tmp_z)

    return slope.img

def maxmin_disk_filter(argsin):

    arr, rad = argsin

    max_arr = filters.generic_filter(arr, nanmax, footprint=disk(rad))
    min_arr = filters.generic_filter(arr, nanmin, footprint=disk(rad))

    return max_arr, min_arr

def spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=500, cutoff_thr=20.,nproc=1):

    # here we assume that the reference DEM is a "clean" post-processed DEM, filtered with QA for low confidence outliers
    # minimum/maximum elevation in circular surroundings based on reference DEM

    ref_arr = ref_dem.img
    res = ref_dem.dx
    rad = int(np.floor(cutoff_kern_size / res))

    if nproc == 1:
        print('Filtering min/max in radius of '+str(cutoff_kern_size)+'m, base threshold of '+str(cutoff_thr)+'m on 1 proc...')
        max_arr, min_arr = maxmin_disk_filter((ref_arr, rad))
    else:
        print('Filtering min/max in radius of '+str(cutoff_kern_size)+'m, base threshold of '+str(cutoff_thr)+'m on '+str(nproc)+' procs...')
        nopt=int(np.floor(np.sqrt(nproc)))
        nblocks=[nopt,nopt]
        patches, inv, split = patchify(ref_arr,nblocks,rad)

        pool = mp.Pool(nproc, maxtasksperchild=1)
        argsin = [(p, rad) for i, p in enumerate(patches)]
        outputs = pool.map(maxmin_disk_filter, argsin, chunksize=1)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        max_arr = unpatchify(np.shape(ref_arr),zip_out[0],inv,split)
        min_arr = unpatchify(np.shape(ref_arr),zip_out[1],inv,split)

    for i in range(ds_arr.shape[0]):
        ds_arr[i, np.logical_or(ds_arr[i, :] > (max_arr + cutoff_thr), ds_arr[i, :] < (min_arr - cutoff_thr))] = np.nan

    return ds_arr

def fit_stack(fn_stack, fit_extent=None, fn_ref_dem=None, ref_dem_date=None, filt_ref='min_max', time_filt_thresh=[-30,5],
              inc_mask=None, gla_mask=None, nproc=1, method='gpr', opt_gpr=False, kernel=None, filt_ls=False,
              conf_filt_ls=0.99, tlim=None, tstep=0.25, outfile='fit.nc', write_filt=False, clobber=False):
    """
    Given a netcdf stack of DEMs, perform temporal fitting with uncertainty propagation

    :param fn_stack: Filename for input netcdf file
    :param fn_ref_dem: Filename for input reference DEM (maybe we change that to a footprint shapefile to respect your original structure?)
    :param ref_dem_date: Date of ref_dem
    :param filt_ref: Type of filtering
    :param time_filt_thresh: Maximum dh/dt from reference DEM for time filtering
    :param inc_mask: Optional inclusion mask
    :param nproc: Number of cores for multiprocessing [1]
    :param method: Fitting method, currently supported: Gaussian Process Regression "gpr", Ordinary Least Squares "ols" and Weighted Least Squares "wls" ["gpr"]
    :param opt_gpr: Run learning optimization in the GPR fitting [False]
    :param kernel: Kernel
    :param filt_ls: Filter least square with a first fit [False]
    :param conf_filt_ls: Confidence interval to filter least square fit [99%]
    :param tstep: Temporal step for fitted stack [0.25 year]
    :param outfile: Path to outfile
    :param write_filt: Write filtered stack to file
    :param clobber: Overwrite existing output files
    :return:
    """

    assert method in ['gpr', 'ols', 'wls'], "Method must be one of gpr, ols or wls."
    print('Reading dataset: ' + fn_stack)

    # we are already chunking manually (split/stitch) below so let's leave chunking aside for now
    # ds = xr.open_dataset(fn_stack, chunks={'x': 100, 'y': 100})

    ds = xr.open_dataset(fn_stack)

    if fit_extent is not None:
        xmin, xmax, ymin, ymax = fit_extent
        ds = ds.sel(x=slice(xmin,xmax),y=slice(ymin,ymax))
    # ds.load()

    print('Original temporal size of stack is '+str(ds.time.size))
    print('Original spatial size of stack is '+str(ds.x.size)+','+str(ds.y.size))

    if inc_mask is not None:
        ds_orig = ds.copy()
        land_mask = get_stack_mask(inc_mask, ds)
        ds, submask, slices = tt.sel_dc(ds,None,land_mask)
        print('Including only '+str(np.count_nonzero(land_mask))+' pixels out of '+ str(np.shape(land_mask)[0]*np.shape(land_mask)[1]) +' on this tile')
        ds_arr = ds.z.values
        ds_arr[:, ~submask] = np.nan
        print('Consequently removing void DEMs...')
        non_void = np.count_nonzero(np.isfinite(ds_arr),axis=(1,2)) > 0
        ds = ds.isel(time=non_void)

    print('Temporal size of stack is now: '+str(ds.time.size))
    print('Spatial size of stack is now: '+str(ds.x.size)+','+str(ds.y.size))

    print('Filtering with max RMSE of 20...')

    keep_vals = ds.uncert.values < 20
    ds = ds.isel(time=keep_vals)

    print('Temporal size of stack is now: '+str(ds.time.size))

    print('Merging ASTER DEMs with exact same date...')

    t_vals = ds.time.values
    dates_rm_dupli = list(set(list(t_vals)))
    ind_firstdate = []
    for i, date in enumerate(dates_rm_dupli):
        ind_firstdate.append(list(t_vals).index(date))
    ind_firstdate = sorted(ind_firstdate)
    ds_filt = ds.isel(time=np.array(ind_firstdate))
    for i in range(len(dates_rm_dupli)):
        t_ind = (t_vals == dates_rm_dupli[i])
        if len(t_ind)>1:
            #careful, np.nansum gives back zero for an axis full of NaNs
            mask_nan = np.all(np.isnan(ds.z.values[t_ind,:]),axis=0)
            ds_filt.z.values[i, :] = np.nansum(ds.z.values[t_ind, :] * 1./ds.uncert.values[t_ind,None,None]**2, axis=0)/np.nansum(1./ds.uncert.values[t_ind,None,None]**2, axis=0)
            ds_filt.z.values[i, mask_nan] = np.nan
            ds_filt.uncert.values[i] = np.nansum(ds.uncert.values[t_ind] * 1./ds.uncert.values[t_ind]**2) / np.nansum(1./ds.uncert.values[t_ind]**2)

    ds = ds_filt

    print('Final temporal size of stack is '+str(ds.time.size))

    ds_arr = ds.z.values
    t_vals = ds.time.values
    uncert = ds.uncert.values

    if gla_mask is not None:
        uns_mask = get_stack_mask(gla_mask, ds)
        uns_arr = uns_mask[np.newaxis,:,:]
    else:
        uns_arr = None

    filt_vals = (t_vals - np.datetime64('2000-01-01')).astype('timedelta64[D]').astype(int)

    # minimum/maximum elevation on Earth
    ds_arr[np.logical_or(ds_arr < -400, ds_arr > 8900)] = np.nan

    if fn_ref_dem is not None:
        assert filt_ref in ['min_max', 'time','both'], "fn_ref must be one of: min_max, time, both"
        tmp_geo = st.make_geoimg(ds)
        tmp_dem = GeoImg(fn_ref_dem)
        ref_dem = tmp_dem.reproject(tmp_geo)
        if filt_ref == 'min_max':
            print('Filtering spatially using min/max values in {}'.format(fn_ref_dem))
            ds_arr = spat_filter_ref(ds_arr, ref_dem, nproc=nproc)
        elif filt_ref == 'time':
            if ref_dem_date is None:
                print('Reference DEM time stamp not specified, defaulting to 01.01.2000')
                ref_dem_date = np.datetime64('2000-01-01')
            print('Filtering temporally with threshold of {} m/a'.format(time_filt_thresh))
            ds_arr = time_filter_ref(ds_arr, t_vals, ref_dem, ref_dem_date, thresh=time_filt_thresh)
        elif filt_ref == 'both':
            print('Filtering spatially using min/max values in {}'.format(fn_ref_dem))
            ds_arr = spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=200, cutoff_thr=200., nproc=nproc)
            ds_arr = spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=500, cutoff_thr=50., nproc=nproc)
            ds_arr = spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=2000, cutoff_thr=20, nproc=int(np.floor(nproc/4)))
            if ref_dem_date is None:
                print('Reference DEM time stamp not specified, defaulting to 01.01.2000')
                ref_dem_date = np.datetime64('2000-01-01')
            print('Filtering temporally with threshold of {} m/a'.format(time_filt_thresh))
            ds_arr = time_filter_ref(ds_arr, t_vals, ref_dem, ref_dem_date, thresh=time_filt_thresh)

    #get median slope from ASTER data + reference DEM if provided (better to avoid having NaN slope on the edge pixels)
    if nproc == 1:
        print('Estimating terrain slope to constrain uncertainties with 1 proc...')
        err_arr = np.zeros(np.shape(ds_arr))
        for i in range(len(t_vals)):
            slope = get_slope(st.make_geoimg(ds,i))
            slope.img[slope.img>70] = np.nan
            err_arr[i,:,:] = slope.img
    else:
        print('Estimating terrain slope to constrain uncertainties with '+str(nproc)+' procs...')
        tmp_img = st.make_geoimg(ds)
        in_met = (tmp_img.gt, tmp_img.proj_wkt, tmp_img.npix_x, tmp_img.npix_y)
        argsin_z = [(ds_arr[i,:],in_met) for i in range(ds.time.size)]
        pool = mp.Pool(nproc, maxtasksperchild=1)
        outputs_z = pool.map(wrapper_slope, argsin_z, chunksize=1)
        pool.close()
        pool.join()
        err_arr = np.stack(outputs_z, axis=0)
    med_slope = np.nanmedian(err_arr,axis=0)

    if fn_ref_dem is not None:
        tmp_dem = GeoImg(fn_ref_dem)
        slope_all = get_slope(tmp_dem)
        slope_ref = slope_all.reproject(st.make_geoimg(ds))
        med_slope[np.isnan(med_slope)] = slope_ref.img[np.isnan(med_slope)]

    # define temporal prediction vector
    if tlim is None:
        y0 = t_vals[0].astype('datetime64[D]').astype(object).year
        y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    else:
        y0 = tlim[0].astype('datetime64[D]').astype(object).year
        y1 = tlim[-1].astype('datetime64[D]').astype(object).year + 1.1
    fit_t = np.arange(y0, y1, tstep) - y0
    nice_fit_t = [np.timedelta64(int(d), 'D').astype(int) for d in np.round(fit_t * 365.2524)]

    print('Fitting with method: ' + method)

    fn_filt = os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(fn_stack))[0]+'_filtered.nc')

    if nproc == 1:
        print('Processing with 1 core...')
        if method == 'gpr':
            out_cube, filt_cube = gpr_wrapper((ds_arr, 0, t_vals, med_slope, uncert, fit_t, opt_gpr, kernel, uns_arr))
            cube_to_stack(ds, out_cube, y0, nice_fit_t, slope_arr=med_slope, outfile=outfile, clobber=clobber)
            if write_filt:
                cube_to_stack(ds, filt_cube, y0, filt_vals, outfile=fn_filt, clobber=clobber, filt_bool=True, ci=False)
        elif method in ['ols', 'wls']:
            if method == 'ols':
                weig = False
            else:
                weig = True
            out_arr, filt_cube = ls_wrapper((ds_arr, 0, t_vals, uncert, weig, filt_ls, conf_filt_ls))
            arr_to_img(ds, out_arr, outfile=outfile)
            if write_filt:
                cube_to_stack(ds, filt_cube, y0, filt_vals, outfile=fn_filt, clobber=clobber, filt_bool=True, ci=False)
    else:
        print('Processing with ' + str(nproc) + ' cores...')
        # now, try to figure out the nicest way to break up the image, given the number of processors
        # there has to be a better way than this... i'll look into it
        # TODO: do the tiling with xarray/dask, and proper chunksizes

        if method in ['ols','wls']:
            #here calculation is done matricially so we want to use all cores with the largest tiles possible
            opt_n_tiles = int(np.floor(np.sqrt(nproc)))
            n_x_tiles = opt_n_tiles
            n_y_tiles = opt_n_tiles
        elif method == 'gpr':
            #here calculation is within a for loop: better to have small tiles to get an idea of the processing speed
            n_x_tiles = np.ceil(ds['x'].shape[0] / 30).astype(int)  # break it into 10x10 tiles
            n_y_tiles = np.ceil(ds['y'].shape[0] / 30).astype(int)

        pool = mp.Pool(nproc, maxtasksperchild=1)
        split_arr = splitter(ds_arr, (n_y_tiles, n_x_tiles))
        split_slp = splitter(med_slope[None, :, :], (n_y_tiles,n_x_tiles))

        if uns_arr is not None:
            split_uns = splitter(uns_arr, (n_y_tiles, n_x_tiles))
        else:
            split_uns = [None] * len(split_arr)

        if method == 'gpr':
            argsin = [(s, i, np.copy(t_vals), split_slp[i], np.copy(uncert) ,np.copy(fit_t), opt_gpr, kernel, split_uns[i]) for i, s in
                      enumerate(split_arr)]
            outputs = pool.map(gpr_wrapper, argsin, chunksize=1)
            pool.close()
            pool.join()

            zip_out = list(zip(*outputs))

            out_cube = stitcher(zip_out[0], (n_y_tiles, n_x_tiles))

            #to write back to full extent even if the dataset was subsetted spatially at the top
            # if inc_mask is not None:
            #     full_out_cube = np.zeros((np.shape(out_cube)[0],ds_orig.y.size,ds_orig.x.size)) * np.nan
            #     full_out_cube[slices[0],slices[1]] = out_cube
            #     out_cube = full_out_cube
            #     ds=ds_orig

            cube_to_stack(ds, out_cube, y0, nice_fit_t, slope_arr=med_slope, outfile=outfile, clobber=clobber)
            if write_filt:
                filt_cube = stitcher(zip_out[1], (n_y_tiles, n_x_tiles))
                # if inc_mask is not None:
                #     full_filt_cube = np.zeros((np.shape(filt_cube)[0], ds_orig.y.size, ds_orig.x.size)) * np.nan
                #     full_filt_cube[slices[0], slices[1]] = filt_cube
                #     filt_cube = full_filt_cube

                cube_to_stack(ds, filt_cube, y0, filt_vals, outfile=fn_filt, clobber=clobber, filt_bool=True, ci=False)

        elif method in ['ols', 'wls']:
            if method == 'ols':
                weig = False
            else:
                weig = True
            argsin = [(s, i, np.copy(t_vals), np.copy(uncert), weig, filt_ls, conf_filt_ls) for i, s in
                      enumerate(split_arr)]
            outputs = pool.map(ls_wrapper, argsin, chunksize=1)
            pool.close()
            pool.join()

            zip_out = list(zip(*outputs))

            out_arr = stitcher(zip_out[0], (n_y_tiles, n_x_tiles))
            arr_to_img(ds, out_arr, outfile=outfile)
            if write_filt:
                filt_cube = stitcher(zip_out[1], (n_y_tiles, n_x_tiles))
                cube_to_stack(ds, filt_cube, y0, filt_vals, outfile=fn_filt, clobber=clobber, filt_bool=True, ci=False)