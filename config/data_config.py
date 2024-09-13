import random
import time
import yaml
import logging
import sys
import os
import pandas as pd
import datetime as dt
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def read_cfg_file(filename):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg


def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

config_filename = '../config/data_config.yaml'

with open(config_filename, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=Loader)

time_slot = cfg['preprocess']['freq']
data_name = cfg['preprocess']['data_name']

root_data_path = cfg['preprocess']['root_data_path']

saved_data_path = os.path.join(cfg['preprocess']['saved_data_path'], data_name)

raw_data_path = os.path.join(root_data_path, cfg['preprocess']['raw_data_path'])
structured_save_dir = os.path.join(saved_data_path, cfg['preprocess']['structured_save_dir'])
train_save_dir = os.path.join(saved_data_path, cfg['preprocess']['train_save_dir'])

raw_od_path = os.path.join(raw_data_path, cfg['preprocess']['raw_od_dir'])
raw_flow_path = os.path.join(raw_data_path, cfg['preprocess']['raw_flow_dir'])
raw_road_path = os.path.join(raw_data_path, cfg['preprocess']['raw_road_dir'])
raw_poi_path = os.path.join(raw_data_path, cfg['preprocess']['raw_poi_dir'])
raw_weather_path = os.path.join(raw_data_path, cfg['preprocess']['raw_weather_dir'])

raw_bike_path = os.path.join(raw_od_path, cfg['preprocess']['raw_bike_dir'])
raw_wyc_path = os.path.join(raw_od_path, cfg['preprocess']['raw_wyc_dir'])
raw_bus_path = os.path.join(raw_od_path, cfg['preprocess']['raw_bus_dir'])

raw_bike_file = os.path.join(raw_bike_path, cfg['preprocess']['raw_bike_file'])
raw_wyc_file = os.path.join(raw_wyc_path, cfg['preprocess']['raw_wyc_file'])
raw_bus_file = os.path.join(raw_bus_path, cfg['preprocess']['raw_bus_file'])

raw_weather_file = os.path.join(raw_weather_path, cfg['preprocess']['raw_weather_file'])
raw_used_poi_path = os.path.join(raw_poi_path, cfg['preprocess']['raw_used_poi_path'])

freq_dir = os.path.join(structured_save_dir, cfg['preprocess']['freq'])

bike_coordinate_path = os.path.join(structured_save_dir, cfg['preprocess']['bike_coordinate'])
bike_order_path = os.path.join(structured_save_dir, cfg['preprocess']['bike_order'])
bike_station_path = os.path.join(structured_save_dir, cfg['preprocess']['bike_station'])

bus_coordinate_path = os.path.join(structured_save_dir, cfg['preprocess']['bus_coordinate'])
bus_order_path = os.path.join(structured_save_dir, cfg['preprocess']['bus_order'])
bus_station_path = os.path.join(structured_save_dir, cfg['preprocess']['bus_station'])

wyc_coordinate_path = os.path.join(structured_save_dir, cfg['preprocess']['wyc_coordinate'])
wyc_order_path = os.path.join(structured_save_dir, cfg['preprocess']['wyc_order'])
wyc_station_path = os.path.join(structured_save_dir, cfg['preprocess']['wyc_station'])


road_data_file = os.path.join(raw_road_path, cfg['preprocess']['road_data_file'])

road_pos_file = os.path.join(raw_road_path, cfg['preprocess']['road_pos_file'])

grid_belong_path = os.path.join(structured_save_dir, cfg['preprocess']['grid_belong'])

road_belong_path = os.path.join(structured_save_dir, cfg['preprocess']['road_belong'])

important_region_path = os.path.join(raw_poi_path, cfg['preprocess']['important_region'])

order_num_thre = cfg['preprocess']['order_num_thre']

start_date = cfg['preprocess']['start_date']
end_date = cfg['preprocess']['end_date']


start_date_ls = pd.date_range(start_date, end_date, freq='1MS',closed='left').strftime("%Y/%m/%d").to_list()
end_date_ls = pd.date_range(start_date, end_date, freq='1MS',closed='right').strftime("%Y/%m/%d").to_list()

xc_holidays = cfg['preprocess']['xc_holidays']
nyc_holidays = cfg['preprocess']['nyc_holidays']

poi_list = cfg['preprocess']['poi_list']



