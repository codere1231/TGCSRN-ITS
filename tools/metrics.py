# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan, mode='dcrnn'):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        if mode == 'dcrnn':
            return np.mean(mae)
        else:
            return np.mean(mae, axis=(0, 1))


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)

        preds = preds[labels > 10]
        mask = mask[labels > 10]
        labels = labels[labels > 10]

        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mae_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    preds = preds[labels > 10]
    mask = mask[labels > 10]
    labels = labels[labels > 10]

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def metric(pred, real):
    mae = masked_mae_torch(pred, real, np.inf).item()
    mape = masked_mape_torch(pred, real, np.inf).item()
    rmse = masked_rmse_torch(pred, real, np.inf).item()
    return mae, mape, rmse


def metric_np(pred, real):
    mae = masked_mae_np(pred, real, np.inf)
    mape = masked_mape_np(pred, real, np.inf)
    rmse = masked_rmse_np(pred, real, np.inf)
    return mae, mape, rmse


def metric_all(preds, reals):
    time = preds[0].shape[2]

    mae = {}
    rmse = {}
    mape = {}

    mae['demand'] = np.zeros(time)
    rmse['demand'] = np.zeros(time)
    mape['demand'] = np.zeros(time)

    if len(preds) > 1:
        for t in range(time):
            mae['demand'][t], mape['demand'][t], rmse['demand'][t] = metric(preds[3][:, :, t, :], reals[3][:, :, t, :])
    else:
        for t in range(time):
            mae['demand'][t], mape['demand'][t], rmse['demand'][t] = metric(preds[0][:, :, t, :], reals[0][:, :, t, :])

    return mae, rmse, mape


def record(all_mae, all_rmse, all_mape, mae, rmse, mape, only_last=False):
    if only_last:
        all_mae['demand'].append(mae['demand'][-1])
        all_rmse['demand'].append(rmse['demand'][-1])
        all_mape['demand'].append(mape['demand'][-1])
    else:
        all_mae['demand'].append(mae['demand'])
        all_rmse['demand'].append(rmse['demand'])
        all_mape['demand'].append(mape['demand'])
