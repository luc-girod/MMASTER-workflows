from __future__ import print_function
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
from pymmaster.stack_tools import create_mmaster_stack
import gdal
from glob import glob
from pymmaster.other_tools import SRTMGL1_naming_to_latlon, latlon_to_UTM, niceextent_utm_latlontile
import pandas as pd

ref_dir = '/calcul/santo/hugonnet/tandem/06_rgi60/'
aster_dir = '/data/icesat/travail_en_cours/romain/data/dems/aster_corr/06_rgi60/'
setsm_dir = '/calcul/santo/hugonnet/setsm/'
out_dir = '/data/icesat/travail_en_cours/romain/data/stacks/06_rgi60/'
ref_gla_csv = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/06_rgi60/cov/list_glacierized_tiles_06_rgi60.csv'
exc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'
tmp_dir = '/calcul/santo/hugonnet/stack/coregs/'
df = pd.read_csv(ref_gla_csv)
# df = df[df['UTM zone']=='26N']
tilelist = df['Tile_name'].tolist()
res = 30
#fix starting date so that fitted stacks and raw stacks have the same timescale: more readable
y0 = 2000

for tile in tilelist:

    lat, lon = SRTMGL1_naming_to_latlon(tile)
    epsg, utm = latlon_to_UTM(lat,lon)

    print('Stacking tile: ' + tile + ' in UTM zone '+utm)

    #reference DEM
    ref_utm_dir = os.path.join(ref_dir,utm)
    ref_vrt = os.path.join(ref_utm_dir,'tmp_'+utm+'.vrt')
    ref_list=glob(os.path.join(ref_utm_dir,'**/*.tif'),recursive=True)
    if not os.path.exists(ref_vrt):
        gdal.BuildVRT(ref_vrt, ref_list, resampleAlg='bilinear')

    #DEMs to stack
    # setsm_tile_dir = os.path.join(setsm_dir,'processed_'+tile.lower())
    flist1 = glob(os.path.join(aster_dir, '**/*_final.zip'), recursive=True)
    # flist2 = glob(os.path.join(setsm_dir,'*.tif'))

    flist = flist1

    outfile = os.path.join(out_dir,utm,tile+'.nc')
    tmp_dir_tile = os.path.join(tmp_dir,tile)

    extent = niceextent_utm_latlontile(tile,utm,res)
    bobformat_extent = [extent[0],extent[2],extent[1],extent[3]]

    print('Nice extent is:')
    print(extent)

    if len(flist)>0:
        nco = create_mmaster_stack(flist, extent=bobformat_extent, epsg=int(epsg), mst_tiles=ref_vrt, outdir=tmp_dir_tile, exc_mask=exc_mask,res=res, outfile=outfile, coreg=False, uncert=True, clobber=True, add_ref=True,latlontile_nodata=tile, filt_mm_corr=True, l1a_zipped=True ,y0=y0)
        nco.close()
    else:
        print('No DEM intersecting tile found. Skipping...')
