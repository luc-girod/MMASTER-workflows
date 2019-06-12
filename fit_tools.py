import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
from itertools import chain
import time
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ
from numba import jit
import gdal
from pybob.GeoImg import GeoImg
from pybob.image_tools import create_mask_from_shapefile
from warnings import filterwarnings
filterwarnings('ignore')


def nmad(data):
    m = np.nanmedian(data)
    return 1.4826 * np.nanmedian(np.abs(data - m))


def get_land_mask(maskshp, ds):
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
    
    
def parse_epsg(wkt):
    return int(''.join(filter(lambda x: x.isdigit(), wkt.split(',')[-1])))
    
    
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

def iterative_fit(time_vals, data_vals, err_vals, time_pred):
    k1 = C(2.0, (1e-2, 1e2)) * RBF(10, (5, 30)) # other kernels to try to add here?
    k2 = C(1.0, (1e-2, 1e2)) * RBF(1, (1, 5))
    k3 = C(10, (1e-3, 1e3)) * RQ(length_scale=30, length_scale_bounds=(30, 1e3))
    kernel = k1 + k2 + k3
    n_out = 1
    niter = 0
    
    num_finite = data_vals.size
    good_vals = np.isfinite(data_vals)
    
    while n_out > 0 and num_finite > 2 and niter < 3:
        # if we remove a linear trend, normalize_y should be false...
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=err_vals[good_vals], normalize_y=False)
        gp.fit(time_vals[good_vals].reshape(-1, 1), data_vals[good_vals].reshape(-1, 1))        
        y_pred, sigma = gp.predict(time_pred.reshape(-1, 1), return_std=True)
        y_, s_ = interp_data(time_pred, y_pred.squeeze(), sigma.squeeze(), time_vals)
        z_score = np.abs(data_vals - y_) / s_
        isin = z_score < 4
        n_out = np.count_nonzero(~isin)
        
        data_vals[~isin] = np.nan
        time_vals[~isin] = np.nan
        err_vals[~isin] = np.nan
        
        good_vals = np.isfinite(data_vals)
        num_finite = np.count_nonzero(good_vals)
        niter += 1

    if num_finite <= 2:
        y_pred = np.nan * np.zeros(time_pred.shape)
        sigma = np.nan * np.zeros(time_pred.shape)
        z_score = np.nan * np.zeros(data_vals.shape)
    return y_pred.squeeze(), sigma.squeeze(), z_score, good_vals

    
def fit_data(data, t_vals, uncert):
    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    t_pred = np.arange(y0, y1, 0.25) - y0    

    if np.count_nonzero(np.isfinite(data)) < 2:
        return np.nan * np.zeros(np.tile(t_pred, 2).shape)
    ftime = t_vals[np.isfinite(data)]
    fdata = data[np.isfinite(data)]
    ferr = uncert[np.isfinite(data)]
    # try to remove a linear fit from the data before we fit, then add it back in when we're done.
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))

    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in ftime])
    t_scale = (ftime_delta / total_delta) * (int(y1) - y0)

    try:
        reg = detrend(t_scale, fdata, ferr)
    except:
        y_pred = np.nan * np.zeros(t_pred.shape)
        sigma = np.nan * np.zeros(t_pred.shape)
        z_score = np.nan * np.zeros(fdata.shape)
        good_vals = np.isfinite(np.nan * np.zeros(fdata.size))
        return y_pred, sigma, z_score, good_vals
    
    fdata = fdata - reg.predict(t_scale.reshape(-1, 1)).squeeze()
    l_trend = reg.predict(t_pred.reshape(-1, 1)).squeeze()
    
    #std_nmad_rat = np.std(fdata) / nmad(fdata) 
    #if std_nmad_rat > 20: 
    #    isout = np.abs(fdata) > 10 * nmad(fdata) 
    #else: 
    #    isout = np.abs(fdata) > 4 * np.std(fdata) 

    #fdata = fdata[~isout]
    #t_scale = t_scale[~isout]
    #ferr = ferr[~isout]

    y_pred, sigma, z_score, good_vals = iterative_fit(t_scale, fdata, ferr, t_pred)
    return y_pred + l_trend, sigma, z_score, good_vals


@jit 
def fitall_jit(argsin): 
    subarr, i, t_vals, uncert, new_t = argsin
    start = time.time()
    Y, X = subarr[0].shape 
    outarr = np.nan * np.zeros((new_t * 2, Y, X)) 
    for x in range(X): 
        for y in range(Y):
            out = fit_data(subarr[:, y, x], t_vals, uncert) 
            outarr[:, y, x] = out
    elaps = time.time() - start
    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr


def splitter(img, nblocks, overlap=0):
    split1 = np.array_split(img, nblocks[0], axis=1)
    split2 = [np.array_split(im, nblocks[1], axis=2) for im in split1]
    olist = [np.copy(a) for a in list(chain.from_iterable(split2))]
    return olist


def stitcher(outputs, nblocks, overlap=0):
    stitched = []
    if np.array(nblocks).size == 1:
        nblocks = np.array([nblocks, nblocks])
    for i in range(nblocks[0]):
        stitched.append(outputs[i*nblocks[1]])
        for j in range(1, nblocks[1]):
            outind = j + i*nblocks[1]
            stitched[i] = np.concatenate((stitched[i], outputs[outind]), axis=2)
    return np.concatenate(tuple(stitched), axis=1)

