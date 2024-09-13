# !/usr/bin/env python
# -*- coding:utf-8 -*-


from datetime import datetime
import torch
import numpy as np
import sys
import os
import argparse
from models.TGCSRN import get_TGCSRN
from ModelTest import baseline_test
from ModelTrain import baseline_train

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from config.config import *
from preprocess.datasets import load_dataset
from tools.utils import sym_adj, asym_adj
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='config_NYC-Traffic.yaml', help='Set the config file')
    args = parser.parse_args()
    config_file = args.yaml
    with open(os.path.join('config/', config_file), 'r', encoding='gbk') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)

    set_seed(cfg['seed'], True)
    expid = cfg['expid']

    current_time_for_path = datetime.now()
    formatted_time_for_path = current_time_for_path.strftime("%Y-%m-%d-%H_%M_%S")
    base_path = cfg['base_path']
    dataset_name = cfg['dataset_name']
    dataset_path = os.path.join(base_path, dataset_name)

    log_path = os.path.join('Result', cfg['model_name'], formatted_time_for_path, cfg['data']['freq'], 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    save_path = os.path.join('Result', cfg['model_name'], formatted_time_for_path, cfg['data']['freq'], 'ckpt')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    log_dir = log_path
    log_level = 'INFO'
    log_name = 'info_' + formatted_time_for_path + '_exp' + str(expid) + '.log'
    logger = get_logger(log_dir, __name__, log_name, level=log_level)

    with open(os.path.join(log_dir, config_file), 'w+') as _f:
        yaml.safe_dump(cfg, _f)

    cfg_str = yaml.dump(cfg, default_flow_style=False, allow_unicode=True)

    logger.info(
        f"Configuration:\n\n|||----- Configuration Start -----|||\n{cfg_str}\n|||----- Configuration End -----|||\n\n")
    logger.info(dataset_path)
    logger.info(log_path)

    torch.set_num_threads(3)
    device = torch.device(cfg['device'])

    # set up the dataset
    dataloader = load_dataset(dataset_path,
                              cfg['data']['train_batch_size'],
                              cfg['data']['val_batch_size'],
                              cfg['data']['test_batch_size'],
                              logger=None
                              )

    try:
        geo_graph = np.load(os.path.join(base_path, 'graph/geo_adj.npy')).astype(np.float32)
        if cfg['model']['norm_graph'] == 'sym':
            norm_geo_graph = torch.tensor(sym_adj(geo_graph)).to(device)
        elif cfg['model']['norm_graph'] == 'asym':
            norm_geo_graph = torch.tensor(asym_adj(geo_graph)).to(device)
        else:
            norm_geo_graph = torch.tensor(geo_graph).to(device)
        static_norm_adjs = [norm_geo_graph]
        logger.info("Graph information found, using graph information")
    except:
        static_norm_adjs = None
        geo_graph = None
        logger.info("No graph information found, using random initialization")

    cluster_num = cfg['data']['poi_cluster']
    model_name = cfg['model_name']

    if cfg['data']['use_external']:
        input_dim = cfg['model']['input_dim'][0] + cfg['model']['context_channels'][0] + \
                    cfg['model']['context_channels'][1]
    else:
        input_dim = cfg['model']['input_dim'][0]

    time_dim = cfg['model']['context_channels'][0]
    hidden_dim = cfg['model']['hidden_dim']
    output_dim = cfg['model']['output_dim']
    num_nodes = cfg['data']['cluster_num']
    num_for_target = cfg['data']['num_for_target']
    num_for_predict = cfg['data']['num_for_predict']
    context_channels = cfg['model']['context_channels']
    context_dims = cfg['model']['context_dims']
    activation_type = cfg['model']['activation_type']
    fuse_type = cfg['model']['fuse_type']
    use_skip = cfg['model']['use_skip']
    use_weather = cfg['model']['use_weather']
    use_adaptive = cfg['model']['use_adaptive']
    node_emb = cfg['model']['node_emb']
    gcn_depth = cfg['model']['gcn_depth']
    d_model_cluster = cfg['model']['d_model_cluster']
    retention_rate = cfg['model']['retention_rate']

    val_mae_list = []
    val_mape_list = []
    val_rmse_list = []
    mae_list = []
    mape_list = []
    rmse_list = []

    for i in range(cfg['runs']):
        if model_name == 'TGCSRN':
            model = get_TGCSRN(in_dim=input_dim,
                               out_dim=output_dim,
                               hidden_dim=hidden_dim,
                               num_for_target=num_for_target,
                               num_for_predict=num_for_predict,
                               num_nodes=num_nodes,
                               context_channels=context_channels,
                               context_dims=context_dims,
                               activation_type=activation_type,
                               supports=static_norm_adjs,
                               region_belong=None,
                               cluster_num=cluster_num,
                               device=device,
                               gcn_depth=gcn_depth,
                               fuse_type=fuse_type,
                               use_skip=use_skip,
                               use_weather=use_weather,
                               use_adaptive=use_adaptive,
                               node_emb=node_emb,
                               d_model=d_model_cluster,
                               num_hours_per_day=48,
                               blocks=2).to(device)
        else:
            logger.error(f"Error: Model is not defined.")
            raise ValueError("Model is not defined.")

        logger.info(model_name)

        if cfg['test_only']:
            val_mae, val_mape, val_rmse, mae, mape, rmse = baseline_test(i,
                                                                         model,
                                                                         dataloader,
                                                                         device,
                                                                         logger,
                                                                         cfg,
                                                                         formatted_time_for_path)
        else:
            val_mae, val_mape, val_rmse, mae, mape, rmse = baseline_train(i,
                                                                          model,
                                                                          cfg['model_name'],
                                                                          dataloader,
                                                                          static_norm_adjs,
                                                                          device,
                                                                          logger,
                                                                          cfg,
                                                                          formatted_time_for_path)

        val_mae_list.append(val_mae)
        val_mape_list.append(val_mape)
        val_rmse_list.append(val_rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

    mae_list = np.array(mae_list)
    mape_list = np.array(mape_list)
    rmse_list = np.array(rmse_list)

    amae = np.mean(mae_list, 0)
    amape = np.mean(mape_list, 0)
    armse = np.mean(rmse_list, 0)

    smae = np.std(mae_list, 0)
    smape = np.std(mape_list, 0)
    srmse = np.std(rmse_list, 0)

    logger.info('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.mean(val_mae_list), np.mean(val_rmse_list), np.mean(val_mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.std(val_mae_list), np.std(val_rmse_list), np.std(val_mape_list)))
    logger.info('\n\n')

    logger.info('Test\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.std(mae_list), np.std(rmse_list), np.std(mape_list)))
    logger.info('\n\n')
