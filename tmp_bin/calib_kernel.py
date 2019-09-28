from __future__ import print_function
import os
import pymmaster.fit_tools as ft
import pandas as pd
import numpy as np
#
# method='wls'
# inc_mask=None
# fn_stack = '/calcul/santo/hugonnet/stack/N46E007.nc'
# fn_ref_dem='/calcul/santo/hugonnet/tandem/11_rgi60/32N/TDX_90m_05hem_N46E007.tif'
outfile='/calcul/santo/hugonnet/stack/N46E007_gprcorr.nc'
# ref_dem_date=np.datetime64('2015-01-01')
# write_filt=True
# nproc=64
# clobber=True
# time_filt_thresh=50.
# tstep=0.25
# opt_gpr=False
# kernel=None
# filt_ref='both'
# filt_ls=False
# conf_filt_ls=0.99
#
# ft.fit_stack(fn_stack,fn_ref_dem=fn_ref_dem,ref_dem_date=ref_dem_date,filt_ref=filt_ref,write_filt=True,outfile=outfile,method=method,filt_ls=filt_ls,conf_filt_ls=conf_filt_ls,nproc=nproc,clobber=True)

fn_stack='/calcul/santo/hugonnet/stack/N46E007_filtered.nc'
shp_mask='/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'

var_csv = '/calcul/santo/hugonnet/stack/var_dh.csv'
fn_dh = os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(outfile))[0]+'_dh.tif')
bin_dh = [0,0.5,1,2,3,4,5]

ft.draw_vgm_by_dh(fn_dh,bin_dh,fn_stack,shp_mask,var_csv)

# lags, vmean, vstd = ft.estimate_vgm(fn_stack,shp_mask=shp_mask,nproc=64,nsamp=10000)
#
# df = pd.DataFrame()
# df = df.assign(lags=lags,vmean=vmean,vstd=vstd)
# df.to_csv(var_csv)




import pandas as pd
in_csv = '/home/atom/ongoing/var_dh.csv'

df = pd.read_csv(in_csv)
df = df[df['id']==df['id'][80]]
ft.plot_vgm(df['lags'],df['vmean'],df['vstd'])