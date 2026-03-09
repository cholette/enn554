import csv
import pandas as pd
from datetime import datetime
import contextily as ctx
import numpy as np
from pyproj import Transformer
from PIL import Image
import math

def levelized_cost_of_electricity(E,CAPEX,OPEX,r=0.08):
    N = len(E)
    assert len(OPEX)==N,  "OPEX, CAPEX, and E must be the same length"
    assert len(CAPEX)==N, "OPEX, CAPEX, and E must be the same length"


    NPV_costs, NPV_gen = 0,0
    for ii in range(N):
        d = 1/(1+r)**ii 
        NPV_costs += d*(CAPEX[ii]+OPEX[ii])
        NPV_gen += d*E[ii]

    return NPV_costs/NPV_gen

def read_TMY(file:str,now_year:int=None):
    if now_year is None:
        print("""
Note: The year in the TMY file is the year the typical day was extracted from. 
The datetimes will thus not be sequential. To obtain sequential datetimes, 
provide a now_year (e.g. the current year).
        """)

    metadata = {}
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        info_headers = reader.__next__()
        info = reader.__next__()
        get_metadata = lambda h: info[info_headers.index(h)]
        for h in info_headers:
            metadata[h] = get_metadata(h)
            try:
                metadata[h] = float(metadata[h])
            except:
                continue
            
    tmy_data = pd.read_csv(file,skiprows=[0,1],
                           usecols=[i for i in range(23)])

    times = []
    for ii in range(tmy_data.shape[0]):
        if now_year is not None:
            # The hear in the TMY file is the year of the typical day. If 
            # now_year is provided, we will use that year instead. 
            yr = [now_year for _ in range(tmy_data.shape[0])]
        else:
            yr = tmy_data.Year
        times += [datetime(yr[ii],tmy_data.Month[ii],tmy_data.Day[ii],
                    tmy_data.Hour[ii],tmy_data.Minute[ii])]
        
    # tmy_data['Datetimes'] = times
    tmy_data.insert(0, 'Datetime', times)

    return tmy_data,metadata

def satellite_image(lat,lon,half_width_m,zoom=19,save_path="data/satellite_image.png"):
    
    # Convert lat/lon to Web Mercator (EPSG:3857)
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    cx, cy = t.transform(lon, lat)

    xmin, xmax = cx - half_width_m, cx + half_width_m
    ymin, ymax = cy - half_width_m, cy + half_width_m

    # --- Download satellite imagery ---
    img, extent = ctx.bounds2img(
        xmin, ymin, xmax, ymax,
        zoom=19,
        source=ctx.providers.Esri.WorldImagery
    )
    # extent = (xmin, xmax, ymin, ymax) in Web Mercator metres
    # img is (H, W, 3) uint8, north-up

    h, w = img.shape[:2]
    metres_per_pixel = (extent[1] - extent[0]) / w
    metres_per_100px = metres_per_pixel * 100

    print(f"Image size:          {w} x {h} px")
    print(f"Metres per pixel:    {metres_per_pixel:.3f}")
    print(f"Metres per 100 px:   {metres_per_100px:.2f}")   # <-- enter this in SAM

    Image.fromarray(img).save(save_path)

    scale_factor = 1 / math.cos(math.radians(lat))
    true_metres_per_pixel = metres_per_pixel / scale_factor
    true_metres_per_100px = true_metres_per_pixel * 100
    print(f"True metres per 100 px: {true_metres_per_100px:.2f}")
    print(f"File saved to: {save_path}")
    