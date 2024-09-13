#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from tqdm import tqdm

from helper import Trainer
from tools.metrics import metric, record


def model_val(runid, engine, dataloader, device, logger, cfg, epoch):
    logger.info('Start validation phase.....')

    val_dataloder = dataloader['val']

    valid_loss_list = []
    valid_outputs_list = []

    valid_pred_mape = {}
    valid_pred_rmse = {}
    valid_pred_mae = {}

    valid_pred_mae['demand'] = []
    valid_pred_rmse['demand'] = []
    valid_pred_mape['demand'] = []

    val_tqdm_loader = tqdm(enumerate(val_dataloder))
    for iter, (demand, pos) in val_tqdm_loader:
        tpos = pos[..., :3]
        weather = pos[..., 3:]
        tpos = tpos.to(device)
        weather = weather.to(device)
        demand = demand.to(device)
        loss, mae, rmse, mape, predict, _ = engine.eval(demand, tpos, weather)
        record(valid_pred_mae, valid_pred_rmse, valid_pred_mape, mae, rmse, mape, only_last=False)
        valid_loss_list.append(loss)
        valid_outputs_list.append(predict)

    mval_loss = np.mean(valid_loss_list)

    mvalid_pred_demand_mae = np.mean(valid_pred_mae['demand'])
    mvalid_pred_demand_mape = np.mean(valid_pred_mape['demand'])
    mvalid_pred_demand_rmse = np.mean(valid_pred_rmse['demand'])

    predicts = torch.cat(valid_outputs_list, dim=0)

    log = 'Epoch: {:03d}, Valid Total Loss: {:.4f}\n' \
          'Valid Pred Demand MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(epoch, mval_loss, mvalid_pred_demand_mae, mvalid_pred_demand_rmse, mvalid_pred_demand_mape))

    return mval_loss, mvalid_pred_demand_mae, mvalid_pred_demand_rmse, mvalid_pred_demand_mape, predicts


def model_test(runid, engine, dataloader, device, logger, cfg, time_for_path, mode='Train'):
    logger.info('Start testing phase.....')
    test_dataloder = dataloader['test']
    engine.model.eval()

    test_loss_list = []
    test_pred_mape = {}
    test_pred_rmse = {}
    test_pred_mae = {}

    test_pred_mae['demand'] = []
    test_pred_rmse['demand'] = []
    test_pred_mape['demand'] = []
    test_outputs_list = []

    test_tqdm_loader = tqdm(enumerate(test_dataloder))
    all_causal_map = []
    all_time_map = []
    for c in range(7):
        all_causal_map.append([])
    for c in range(96):
        all_time_map.append([])

    for iter, (demand, pos) in test_tqdm_loader:

        tpos = pos[..., :3]
        weather = pos[..., 3:]

        tpos = tpos.to(device)
        weather = weather.to(device)
        demand = demand.to(device)
        if cfg['test_only']:
            pre_training_mat = torch.load(cfg['train']['best_mat'], map_location=device)
            loss, gen_mae, gen_rmse, gen_mape, predict, soft_mat = engine.eval(demand, tpos, weather,
                                                                               pre_training_mat=pre_training_mat,
                                                                               mat_mode='test_only')
        else:
            loss, gen_mae, gen_rmse, gen_mape, predict, soft_mat = engine.eval(demand, tpos, weather,
                                                                               pre_training_mat=None,
                                                                               mat_mode='train')

        test_loss_list.append(loss)
        record(test_pred_mae, test_pred_rmse, test_pred_mape, gen_mae, gen_rmse, gen_mape, only_last=False)
        test_outputs_list.append(predict)

    mtest_loss = np.mean(test_loss_list)
    mtest_pred_demand_mae = np.mean(test_pred_mae['demand'])
    mtest_pred_demand_mape = np.mean(test_pred_mape['demand'])
    mtest_pred_demand_rmse = np.mean(test_pred_rmse['demand'])

    predicts = torch.cat(test_outputs_list, dim=0)
    log = 'Test Total Loss: {:.4f}\n' \
          'Test Pred Demand MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(mtest_loss,
                           mtest_pred_demand_mae, mtest_pred_demand_rmse, mtest_pred_demand_mape, ))

    if mode == 'Test':
        pred_all = predicts.cpu()
        path_save_pred = os.path.join('Result', cfg['model_name'], time_for_path, cfg['data']['freq'], 'result_pred')
        if not os.path.exists(path_save_pred):
            os.makedirs(path_save_pred, exist_ok=True)
        name = 'exp{:s}_Test_mae:{:.4f}_rmse:{:.4f}_mape:{:.4f}'. \
            format(cfg['model_name'], mtest_pred_demand_mae, mtest_pred_demand_rmse, mtest_pred_demand_mape)
        path = os.path.join(path_save_pred, name)
        np.save(path, pred_all)
        logger.info('result of prediction has been saved, path: {}'.format(path))
        logger.info('shape: ' + str(pred_all.shape))

    return (mtest_loss, mtest_pred_demand_mae, mtest_pred_demand_rmse, mtest_pred_demand_mape, predicts, soft_mat)


def baseline_test(runid, model, dataloader, device, logger, cfg, time_for_path):
    demand_scalar = dataloader['scalar_taxi']

    engine = Trainer(
        model,
        base_lr=cfg['train']['base_lr'],
        weight_decay=cfg['train']['weight_decay'],
        milestones=cfg['train']['milestones'],
        lr_decay_ratio=cfg['train']['lr_decay_ratio'],
        min_learning_rate=cfg['train']['min_learning_rate'],
        max_grad_norm=cfg['train']['max_grad_norm'],
        num_for_target=cfg['data']['num_for_target'],
        num_for_predict=cfg['data']['num_for_predict'],
        scaler=demand_scalar,
        device=device
    )

    best_mode_path = cfg['train']['best_mode']
    logger.info("loading {}".format(best_mode_path))

    save_dict = torch.load(best_mode_path, map_location=device)
    engine.model.load_state_dict(save_dict['model_state_dict'])
    logger.info('model load success! {}'.format(best_mode_path))

    total_param = 0
    logger.info('Net\'s state_dict:')
    for param_tensor in engine.model.state_dict():
        logger.info(param_tensor + '\t' + str(engine.model.state_dict()[param_tensor].size()))
        total_param += np.prod(engine.model.state_dict()[param_tensor].size())
    logger.info('Net\'s total params:{:d}'.format(int(total_param)))

    logger.info('Optimizer\'s state_dict:')
    for var_name in engine.optimizer.state_dict():
        logger.info(var_name + '\t' + str(engine.optimizer.state_dict()[var_name]))

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info('Number of model parameters is {:d}'.format(int(nParams)))

    mtest_loss, mtest_mae, mtest_rmse, mtest_mape, predicts, soft_mat = model_test(runid, engine, dataloader,
                                                                                   device,
                                                                                   logger,
                                                                                   cfg, time_for_path,
                                                                                   mode='Test')

    output_save_best_mat = os.path.join('Result', cfg['model_name'], time_for_path, cfg['data']['freq'], 'mat')
    if not os.path.exists(output_save_best_mat):
        os.makedirs(output_save_best_mat)
    output_save_best_mat = os.path.join(output_save_best_mat, 'save_best_mat_test.pth')
    if os.path.exists(output_save_best_mat):
        raise FileExistsError(f"File exists! {output_save_best_mat}")
    torch.save(soft_mat, output_save_best_mat)
    return mtest_mae, mtest_mape, mtest_rmse, mtest_mae, mtest_mape, mtest_rmse
