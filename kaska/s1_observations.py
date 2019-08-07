#!/usr/bin/env python
"""Dealing with Sentinel 1 observations"""


import datetime as dt
import logging
from collections import namedtuple
from pathlib import Path

import gdal
import numpy as np

from utils import reproject_data, define_temporal_grid

gdal.UseExceptions()

LOG = logging.getLogger(__name__ + ".Sentinel1_Observations")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - ' +
                                  '%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    LOG.addHandler(ch)
LOG.propagate = False
# A SIAC data storage type
S1data = namedtuple(
    #"S1data", "time observations uncertainty mask metadata emulator"
    "S1data", "time VV VH theta VV_unc VH_unc"
)


layers = [
    "sigma0_vv_norm_multi_db",
    "sigma0_vh_norm_multi_db",
    "localIncidenceAngle"
    ]


def get_s1_dates(s1_file):
    """Gets the dates from a LMU processed netCDF Sentinel 1 file"""
    times = [float(s1_file.GetRasterBand(b+1).GetMetadata()['NETCDF_DIM_time']) 
                                for b in range(s1_file.RasterCount)]
    times = [dt.datetime(1970,1,1) + dt.timedelta(days=x) for x in times ]
    LOG.info(f"Sentinel 1 First obs: {times[0].strftime('%Y-%m-%d'):s}")
    LOG.info(f"Sentinel 1 Last obs: {times[-1].strftime('%Y-%m-%d'):s}")
    return times


class Sentinel1Observations(object):
    def __init__(
        self,
        netCDF_file,
        state_mask,
        chunk=None,
        time_grid=None,
        nc_layers = {"VV": "sigma0_vv_norm_multi_db",
                    "VH": "sigma0_vh_norm_multi_db",
                    "theta": "localIncidenceAngle"}
    ):
        self.time_grid = time_grid
        self.state_mask = state_mask
        self.nc_file = Path(netCDF_file)
        self.s1_data_ptr = {}
        for layer, layer_name in nc_layers.items():
            fname = f'NETCDF:"{self.nc_file.as_posix():s}":{layer_name:s}'
            self.s1_data_ptr[layer] = reproject_data(fname, output_format="VRT", 
                                                  srcSRS="EPSG:4326",
                                                  target_img=self.state_mask)
        s1_dates = get_s1_dates(self.s1_data_ptr[layer])
        self.dates = {x:(i+1) 
                            for i, x in enumerate(s1_dates) 
                            if ( (x >= self.time_grid[0]) and 
                            (x <= self.time_grid[-1]))}

    def read_time_series(self, time_grid):
        
        early = time_grid[0]
        late = time_grid[-1]
        
        sel_dates = [k for k,v in self.dates.items()
                           if ((k >= early) and (k < late))]
        sel_bands = [v for k,v in self.dates.items()
                           if ((k >= early) and (k < late))]
        obs = {}
        for ii, layer in enumerate(self.s1_data_ptr.keys()):
                obs[layer] = np.array([self.s1_data_ptr[
                                       layer].GetRasterBand(i).ReadAsArray()
                                         for i in sel_bands])
        the_obs = S1data(sel_dates, obs['VV'], obs['VH'], obs['theta'], 0.5, 0.5)
        return the_obs
    
if __name__ == "__main__":
    start_date = dt.datetime(2017, 3, 1)
    end_date = dt.datetime(2017, 9, 1)
    temporal_grid_space = 5
    temporal_grid = define_temporal_grid(start_date, end_date,
                                        temporal_grid_space)
    nc_file = "/data/selene/ucfajlg/ELBARA_LMU/mirror_ftp/141.84.52.201/S1/S1_LMU_site_2017_new.nc"
    s1_obs = Sentinel1Observations(nc_file,
                "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
                time_grid=temporal_grid)
    vv = s1_obs.read_time_series(temporal_grid[:5])