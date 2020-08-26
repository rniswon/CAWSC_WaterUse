import GridMETDataCollector as gmet
import time
import os

if __name__ == '__main__':
    # start_time = time.time()

    # input shapefile folder
    # wgs84 shapefiles
    shpFolder = os.path.join('GIS', 'WaterUseAreas', 'WGS84')
    ohioZones = os.path.join(shpFolder, 'OH_WGS84.shp')
    # field with the zone names
    zonesShp = os.path.join(shpFolder, 'GU_ALL_FROM_AYMAN_WGS84.shp')
    zoneField = 'GNIS_ID'
    ohioZoneField = 'PLANT_ID'

    # climate types to be processed
    climateFilter = ['etr', 'pet', 'pr', 'sph', 'srad', 'tmmn', 'tmmx', 'vs']

    # filed filter # 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    filterField = {'HUC2': '13'}

    # intialize the data collector
    gmetDC = gmet.DataCollector('GIS')

    # process a single shapefile
    s = time.time()
    test = gmetDC.get_data(zonesShp, zoneField, climate_filter=climateFilter,
                    year_filter='2000', multiprocessing=True, filter_field=filterField,
                    chunksize=None, save_to_csv=False)
    print('time', time.time() - s)
    print('DONE')

    print(test)

    # loop through mutlipel shapefiles and process
    # shps = {'OH_WGS84.shp': 'PLANT_ID', 'ABCWUA_WGS84.shp': 'CN'}
    # for shp, zoneField in shps.items():

        # zoneShp = os.path.join(shpFolder, shp)

        # climateData = gmetDC.get_data(zoneShp, zoneField, climate_filter=['pet'],
                    # year_filter='2000-2015', multiprocessing=True)