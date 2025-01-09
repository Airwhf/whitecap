import os
import numpy as np
import xarray as xr
import regionmask
from pyhdf.SD import SD, SDC


if __name__ == "__main__":
    
    # File path.
    file_path = '/Volumes/project/github_whitecap/MOD09CMG.A2025006.061.2025008100334.hdf'

    # Open the MODIS file.
    hdf = SD(file_path, SDC.READ)

    # List all datasets.
    datasets = hdf.datasets()
    print("Here lists the all dataset name:")
    for dataset_name in datasets:
        print(dataset_name)
    print('')    

    # Parameter configuration.
    t_min = 0.70
    t_cus = 0.75
    t_max = 0.80
    rwc = 0.55
    
    # Read the dataset for calculation.
    ref_band2_name = 'Coarse Resolution Surface Reflectance Band 2'
    theta_0_name = 'Coarse Resolution Solar Zenith Angle'
    theta_name = 'Coarse Resolution View Zenith Angle'
    psi_name = 'Coarse Resolution Relative Azimuth Angle'
    
    ref_band2 = hdf.select(ref_band2_name)[:] * 1E-4
    theta_0 = hdf.select(theta_0_name)[:] * 0.01
    theta = hdf.select(theta_name)[:] * 0.01
    psi = hdf.select(psi_name)[:] * 0.01
    
    # Create the theta_m mask array.
    cos_theta_m = np.cos(theta_0)*np.cos(theta) - np.sin(theta_0)*np.sin(theta)*np.cos(psi)
    cos_40_deg = 0.6981317007977318
    mask_theta_m = np.where((cos_theta_m < cos_40_deg) & (cos_theta_m > 0), True, False)
    
    # Obtain the latitude and longitude.
    lat = np.linspace(-90, 90, 3600)
    lon = np.linspace(-179.5, 179.5, 7200)
    
    # Mask the land.
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(lon, lat)
    
    # Calculate the whitecap coverage.
    record_wc = []
    for t in [t_min, t_cus, t_max]:
        w = ref_band2/(t*rwc)
        w[~mask_theta_m] = np.nan
        w[land == 0] = np.nan
        w = np.where((w>=0) & (w<=1), w, np.nan)
        record_wc.append(w)
        
    # Create the xarray dataset to store the results.
    ds = xr.Dataset(
        {
            'wc_t_min': (['lat', 'lon'], record_wc[0]),
            'wc_t_max': (['lat', 'lon'], record_wc[2]),
            'wc_t_cus': (['lat', 'lon'], record_wc[1]),
        },
        coords={
            'lat': lat,
            'lon': lon,
        },
    )
    
    output_name = os.path.basename(file_path)
    ds.to_netcdf(output_name, format='NETCDF4_CLASSIC')
    print('Save the results to {}'.format(output_name))
    
        
    

    
