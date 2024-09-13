import pandas as pd
import numpy as np
from tqdm import tqdm
import shapely.geometry
from shapely import wkt
from config.data_config import *
import geopandas
import math

def Intersect(l1, l2):
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]

    temp = l1
    l1 = l2
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]

    if a*b < 0 and c*d < 0:
        return True
    else:
        return False

def getDistance(point1, point2):

    lat1, lng1, lat2, lng2 = point1[0], point1[1], point2[0], point2[1]
    def rad(d):
        return d * math.pi / 180.0
    EARTH_REDIUS = 6378.137
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(pow(math.sin(a / 2), 2) +
                                math.cos(radLat1) *
                                math.cos(radLat2) *
                                pow(math.sin(b / 2), 2)))
    s = s * EARTH_REDIUS
    return s

def get_timestamp(demand: pd.DataFrame, holidays, start_date, end_date, freq):

    time_date = pd.date_range(start_date, end_date, freq=freq, closed='left')

    time = pd.DataFrame(index=time_date)
    time['dayofweek'] = time_date.weekday
    time['timeofday'] = (time_date.hour * 3600 + time_date.minute * 60 + time_date.second) // (10 * 60)
    time['holidays'] = time.index.strftime('%Y-%m-%d').isin(holidays).astype(int)

    return time

def construct_weather_data(weather_data_file,weather_data_path,start_date,end_date):

    weather_data = pd.read_csv(weather_data_file,header=0,encoding='gb2312')
    weather_data['batch_time'] = pd.to_datetime(weather_data['batch_time'], unit='s') + pd.Timedelta('8h')
    weather_data = weather_data.sort_values(by='batch_time')
    weather_data = weather_data.reset_index(drop=True)
    time_range = pd.date_range(start='2021/1/1', end='2022/1/1', freq=time_slot, closed='left')
    weather_data = weather_data[weather_data['batch_time'].isin(time_range)]
    weather_data.index = weather_data['batch_time']
    weather_data = weather_data.reindex(time_range).fillna(method='ffill')

    weather_data['batch_time'] = weather_data.index

    weather_data = weather_data[['batch_time', 'condition_id', 'temperature', 'wind_speed', 'humidity', 'aqi']]
    weather_data = pd.concat([weather_data, pd.get_dummies(weather_data['condition_id'])],axis=1)
    weather_data = weather_data.drop(columns=['condition_id'])

    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)

    weather_data[['temperature', 'wind_speed', 'humidity', 'aqi']] = weather_data[['temperature',
                                                'wind_speed', 'humidity', 'aqi']].apply(z_scaler)

    weather_data = weather_data[(weather_data['batch_time']>=start_date)&
                                (weather_data['batch_time']<end_date)]
    weather_data.to_hdf(os.path.join(weather_data_path,'weather.h5'),key='weather',mode='w')

def construct_poi_data(poi_data_path, data_path, region_info):
    poi_data = pd.DataFrame()
    for name in sorted(os.listdir(poi_data_path)):
        other = pd.read_excel(os.path.join(poi_data_path, name), header=0)
        if not other.empty:
            poi_data = pd.concat([poi_data, other], axis=0, ignore_index=True)

    poi_data = poi_data[poi_data['mid_cate'].isin(poi_list)]

    poi_data['belong'] = 9999


    for row_1 in tqdm(poi_data.itertuples()):
        for row_2 in region_info.itertuples():
            point = shapely.geometry.Point(row_1.lng, row_1.lat)
            if point.within(row_2.geometry):
                    poi_data.loc[row_1.Index, 'belong'] = row_2.RID


    tmp = poi_data[poi_data['belong']!=9999].groupby(['big_cate', 'belong'])['count'].sum()
    region_poi = tmp.unstack(level='belong', fill_value=0).T

    print(region_poi.shape[0])
    full_index = pd.Index([i for i in range(len(region_info))])

    region_poi = region_poi.reindex(full_index).fillna(0)
    region_poi.to_hdf(os.path.join(data_path,'poi.h5'), key='poi', mode='w')

def construct_poi_data_NYC(poi_data_path, data_path, region_info):
    poi_file = os.path.join(poi_data_path,'gis_osm_pois_free_1.shp')
    poi_info = geopandas.GeoDataFrame.from_file(poi_file, header=0)
    poi_data = poi_info[poi_info['fclass'].isin(poi_list)]

    poi_data['belong'] = 9999


    for row_1 in tqdm(poi_data.itertuples()):
        for row_2 in region_info.itertuples():
            point = row_1.geometry
            if point.within(row_2.geometry):
                    poi_data.loc[row_1.Index, 'belong'] = row_2.RID
    poi_data = pd.DataFrame(poi_data)
    poi_data = poi_data[['fclass', 'belong']]
    poi_data['num'] = 1
    tmp = poi_data[poi_data['belong']!=9999].groupby(['fclass', 'belong'])['num'].sum()
    region_poi = tmp.unstack(level='belong', fill_value=0).T
    print(region_poi.shape[0])
    print(len(region_info))
    full_index = region_info.index
    region_poi = region_poi.reindex(full_index).fillna(0)
    region_poi.to_hdf(os.path.join(data_path,'poi.h5'), key='poi', mode='w')





def generate_od_graph(grid_belong: pd.DataFrame, taxi_data: pd.DataFrame):
    taxi_node_list = grid_belong.station
    grid_belong_copy = grid_belong.copy()
    grid_belong_copy = grid_belong_copy.set_index('station')
    taxi_data = taxi_data[(taxi_data.start_station.isin(taxi_node_list))&
                            (taxi_data.end_station.isin(taxi_node_list))]

    if data_name == 'BJ':
        taxi_data = taxi_data[taxi_data['start_datetime'].dt.month < 7]
    elif data_name == 'NYC':
        taxi_data = taxi_data[taxi_data['start_datetime'].dt.month < 5]
    morning_rush_hours = taxi_data[(taxi_data['start_datetime'].dt.weekday < 5) &
                               (taxi_data['start_datetime'].dt.hour >= 6) &
                               (taxi_data['start_datetime'].dt.hour < 11)&
                                   (taxi_data['end_datetime'].dt.weekday < 5) &
                                   (taxi_data['end_datetime'].dt.hour >= 6) &
                                   (taxi_data['end_datetime'].dt.hour < 11)
                                   ]
    working_hours = taxi_data[(taxi_data['start_datetime'].dt.weekday < 5) &
                          (taxi_data['start_datetime'].dt.hour >= 11) &
                          (taxi_data['start_datetime'].dt.hour < 17)&
                              (taxi_data['end_datetime'].dt.weekday < 5) &
                              (taxi_data['end_datetime'].dt.hour >= 11) &
                              (taxi_data['end_datetime'].dt.hour < 17)
                              ]


    evening_rush_hours = taxi_data[(taxi_data['start_datetime'].dt.weekday < 5) &
                               (taxi_data['start_datetime'].dt.hour >= 17) &
                               (taxi_data['start_datetime'].dt.hour < 22)&
                                   (taxi_data['end_datetime'].dt.weekday < 5) &
                                   (taxi_data['end_datetime'].dt.hour >= 17) &
                                   (taxi_data['end_datetime'].dt.hour < 22)
                                   ]
    weekday_rest_hour = taxi_data[(taxi_data['start_datetime'].dt.weekday < 5) &
                                  ((taxi_data['start_datetime'].dt.hour >= 22) |
                               (taxi_data['start_datetime'].dt.hour < 6))&
                                   (taxi_data['end_datetime'].dt.weekday < 5) &
                                  ((taxi_data['end_datetime'].dt.hour >= 22) |
                                   (taxi_data['end_datetime'].dt.hour < 6))
                                   ]
    weekend_trip_hours = taxi_data[(taxi_data['start_datetime'].dt.weekday >= 5) &
                               (taxi_data['start_datetime'].dt.hour >= 9) &
                               (taxi_data['start_datetime'].dt.hour < 17)&
                                   (taxi_data['end_datetime'].dt.weekday >= 5) &
                                   (taxi_data['end_datetime'].dt.hour >= 9) &
                                   (taxi_data['end_datetime'].dt.hour < 17)
                                   ]
    weekend_evening_hours = taxi_data[(taxi_data['start_datetime'].dt.weekday >= 5) &
                               (taxi_data['start_datetime'].dt.hour >= 17) &
                               (taxi_data['start_datetime'].dt.hour < 22) &
                                      (taxi_data['end_datetime'].dt.weekday >= 5) &
                                      (taxi_data['end_datetime'].dt.hour >= 17) &
                                      (taxi_data['end_datetime'].dt.hour < 22)
                                      ]

    weekend_rest_hour = taxi_data[(taxi_data['start_datetime'].dt.weekday >= 5) &
                                  ((taxi_data['start_datetime'].dt.hour >= 22) |
                               (taxi_data['start_datetime'].dt.hour < 9))&
                                   (taxi_data['end_datetime'].dt.weekday >= 5) &
                                  ((taxi_data['end_datetime'].dt.hour >= 22) |
                                   (taxi_data['end_datetime'].dt.hour < 9))
                                   ]

    all_periods = [morning_rush_hours,working_hours,evening_rush_hours,weekday_rest_hour,
                   weekend_trip_hours,weekend_evening_hours,weekend_rest_hour]
    for i in range(len(all_periods)):
        tmp_cnt_transform = all_periods[i].groupby(['start_station', 'end_station']).size()
        taxi_graph = tmp_cnt_transform.unstack(level='start_station', fill_value=0)
        taxi_graph = taxi_graph.reindex(index=taxi_node_list.values, columns=taxi_node_list.values, fill_value=0)
        taxi_graph = pd.concat([taxi_graph, grid_belong_copy['belong']], axis=1).groupby(['belong']).sum().T
        taxi_graph = pd.concat([taxi_graph, grid_belong_copy['belong']], axis=1).groupby(['belong']).sum().T
        cur_graph = taxi_graph.values
        Zsum=cur_graph.sum(axis=0)
        cur_graph = cur_graph/Zsum
        np.save(os.path.join(train_save_dir,'od_adj_'+str(i)+'.npy'), np.array(cur_graph))
        print(cur_graph.shape)
        pass
    pass

def generate_distance_graph(cluster_info: pd.DataFrame, dis_thre=2):
    cluster_num = cluster_info.shape[0]
    spatial_dis_matrix = np.zeros([cluster_num, cluster_num])

    for i in range(cluster_num):
        for j in range(i + 1, cluster_num):
            dis = getDistance(cluster_info.iloc[i][['latitude', 'longitude']].values,
                              cluster_info.iloc[j][['latitude', 'longitude']].values)
            spatial_dis_matrix[i][j] = spatial_dis_matrix[j][i] = dis

    adj_matrix = np.where(spatial_dis_matrix < dis_thre, 1, 0)

    thre_affinity_matrix = np.where(spatial_dis_matrix < dis_thre, spatial_dis_matrix, np.inf)
    affinity_matrix = np.exp(- 0.08 * thre_affinity_matrix ** 2)
    print('Density of the distance graph: ', str(adj_matrix.sum() / (adj_matrix.shape[0] ** 2)))

    np.save(os.path.join(train_save_dir,'geo_adj'+'.npy'), np.array(adj_matrix))
    np.save(os.path.join(train_save_dir,'geo_affinity'+'.npy'), np.array(affinity_matrix))
