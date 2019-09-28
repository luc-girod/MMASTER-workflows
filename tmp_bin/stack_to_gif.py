from __future__ import print_function
import xarray as xr
import pymmaster.fit_tools as ft
import numpy as np

fn_stack='/calcul/santo/hugonnet/stack/N46E007_final.nc'
out_gif ='/calcul/santo/hugonnet/stack/N46E007.gif'
out_gif_ci = '/calcul/santo/hugonnet/stack/N46E007_ci.gif'

ds = xr.open_dataset(fn_stack)

t0=np.datetime64('2000-09-01')

fig, ims = ft.make_dh_animation(ds,t0=t0,month_a_year=None,dh_max=50,var='z')
ft.write_animation(fig, ims, outfilename=out_gif)

fig, ims = ft.make_dh_animation(ds,month_a_year=None,dh_max=15,var='z_ci')
ft.write_animation(fig, ims, outfilename=out_gif_ci)
