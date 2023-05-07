import logging
import numpy as np
import torch
from tqdm import tqdm
from utils.my_utils import save_model, load_model, EarlyStopMonitor, Metric
import time
from model.CTCP import CTCP
import math
from utils.data_processing import Data
from typing import Tuple, Dict, Type
from torch.nn.modules.loss import _Loss


def select_label(labels, types):
    train_idx = (labels != -1) & (types == 1)
    val_idx = (labels != -1) & (types == 2)
    test_idx = (labels != -1) & (types == 3)
    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


def move_to_device(device, *args):
    results = []
    for arg in args:
        if type(arg) is torch.Tensor:
            results.append(arg.to(dtype=torch.float, device=device))
        else:
            results.append(torch.tensor(arg, device=device, dtype=torch.float))
    return results


def eval_model(model: CTCP, eval: Data, device: torch.device, param: Dict, metric: Metric,
               loss_criteria: _Loss, move_final: bool = False) -> Dict:
    model.eval()
    model.reset_state()
    metric.fresh()
    epoch_metric = {}
    loss = {'train': [], 'val': [], 'test': []}
    with torch.no_grad():
        for x, label in tqdm(eval.loader(param['bs']), total=math.ceil(eval.length / param['bs']), desc='eval_or_test'):
            src, dst, trans_cas, trans_time, pub_time, types = x
            index_dict = select_label(label, types)
            target_idx = index_dict['train'] | index_dict['val'] | index_dict['test']
            trans_time, pub_time, label = move_to_device(device, trans_time, pub_time, label)
            pred = model.forward(src, dst, trans_cas, trans_time, pub_time, target_idx)
            for dtype in ['train', 'val', 'test']:
                idx = index_dict[dtype]
                if sum(idx) > 0:
                    m_target = trans_cas[idx]
                    m_label = label[idx]
                    m_label[m_label < 1] = 1
                    m_label = torch.log2(m_label)
                    m_pred = pred[idx]
                    loss[dtype].append(loss_criteria(m_pred, m_label).item())
                    metric.update(target=m_target, pred=m_pred.cpu().numpy(), label=m_label.cpu().numpy(), dtype=dtype)
            model.update_state()
        for dtype in ['train', 'val', 'test']:
            epoch_metric[dtype] = metric.calculate_metric(dtype, move_history=True, move_final=move_final,
                                                          loss=np.mean(loss[dtype]))
        return epoch_metric


def train_model(num: int, dataset: Data, model: CTCP, logger: logging.Logger, early_stopper: EarlyStopMonitor,
                device: torch.device, param: Dict, metric: Metric, result: Dict):
    train, val, test = dataset, dataset, dataset
    model = model.to(device)
    logger.info('Start training citation')
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    loss_criterion = torch.nn.MSELoss()
    for epoch in range(param['epoch']):
        model.reset_state()
        model.train()
        logger.info(f'Epoch {epoch}:')
        epoch_start = time.time()
        train_loss = []
        for x, label in tqdm(train.loader(param['bs']), total=math.ceil(train.length / param['bs']),
                             desc='training'):
            src, dst, trans_cas, trans_time, pub_time, types = x
            idx_dict = select_label(label, types)
            target_idx = idx_dict['train']
            trans_time, pub_time, label = move_to_device(device, trans_time, pub_time, label)
            pred = model.forward(src, dst, trans_cas, trans_time, pub_time, target_idx)
            if sum(target_idx) > 0:
                target, target_label, target_time = trans_cas[target_idx], label[target_idx], trans_time[target_idx]
                target_label[target_label < 1] = 1
                target_label = torch.log2(target_label)
                target_pred = pred[target_idx]
                optimizer.zero_grad()
                loss = loss_criterion(target_pred, target_label)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            model.update_state()
            model.detach_state()
        epoch_end = time.time()
        epoch_metric = eval_model(model, val, device, param, metric, loss_criterion, move_final=False)
        logger.info(f"Epoch{epoch}: time_cost:{epoch_end - epoch_start} train_loss:{np.mean(train_loss)}")
        for dtype in ['train', 'val', 'test']:
            metric.info(dtype)
        if early_stopper.early_stop_check(epoch_metric['val']['msle']):
            break
        else:
            ...
    logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    load_model(model, param['model_path'], num)
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    final_metric = eval_model(model, test, device, param, metric, loss_criterion, move_final=True)
    logger.info(f'Runs:{num}\n {metric.history}')
    metric.save()
    save_model(model, param['model_path'], num)

    result['msle'] = np.round(result['msle'] + final_metric['test']['msle'] / param['run'], 4)
    result['mape'] = np.round(result['mape'] + final_metric['test']['mape'] / param['run'], 4)
    result['male'] = np.round(result['male'] + final_metric['test']['male'] / param['run'], 4)
    result['pcc'] = np.round(result['pcc'] + final_metric['test']['pcc'] / param['run'], 4)
