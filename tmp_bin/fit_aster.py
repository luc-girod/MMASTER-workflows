from __future__ import print_function
from pymmaster.fit_tools import fit_stack
from glob import glob
import os
import numpy as np
import gdal

method='gpr'
inc_mask=None
# subspat = [383000,400000,5106200,5094000]
subspat = None
ref_dem_date=np.datetime64('2015-01-01')
gla_mask = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'
write_filt=True
nproc=30
clobber=True
tstep=0.25
opt_gpr=False
kernel=None
filt_ref='both'
filt_ls=False
conf_filt_ls=0.99
#specify the exact temporal extent needed to be able to merge neighbouring stacks properly
tlim=[np.datetime64('2000-01-01'),np.datetime64('2019-01-01')]

dir_raw_stacks='/data/icesat/travail_en_cours/romain/data/stacks/04_rgi60/17N/raw/'
utm='17N'
ref_utm_dir = '/calcul/santo/hugonnet/tandem/04_rgi60/17N/'
dir_fit_stacks = '/data/icesat/travail_en_cours/romain/data/stacks/04_rgi60/17N/fit/'
ref_vrt = os.path.join(ref_utm_dir,'tmp_'+utm+'.vrt')
ref_list=glob(os.path.join(ref_utm_dir,'**/*.tif'),recursive=True)
if not os.path.exists(ref_vrt):
    gdal.BuildVRT(ref_vrt, ref_list, resampleAlg='bilinear')

# list_fn_stack = glob(os.path.join(dir_raw_stacks,'*.nc'),recursive=True)

tilelist = ['N71W079','N71W080','N72W079','N72W080','N72W081']
list_fn_stack = [os.path.join(dir_raw_stacks,tile+'.nc') for tile in tilelist]

for fn_stack in list_fn_stack:

    tile = os.path.splitext(os.path.basename(fn_stack))[0]
    outfile=os.path.join(dir_fit_stacks,tile+'_final.nc')

    fit_stack(fn_stack,fit_extent=subspat,fn_ref_dem=ref_vrt,ref_dem_date=ref_dem_date,tstep=tstep,tlim=tlim,gla_mask=gla_mask,filt_ref=filt_ref,time_filt_thresh=[-30,5],write_filt=True,outfile=outfile,method=method,filt_ls=filt_ls,conf_filt_ls=conf_filt_ls,nproc=nproc,clobber=True)

print('Fin.')