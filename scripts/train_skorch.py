from skorch import NeuralNet
import skorch.callbacks.training as training
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

import os
import datetime
import json

from src.models import ActTime
from configs.option import parse_args

from dask.distributed import Client
from joblib import parallel_backend


def get_datasets(model_name):
    train_data_dir = f"datasets/{model_name}/train"

    low_res_signals = torch.from_numpy(
        np.loadtxt(f"{train_data_dir}/low_res.txt", delimiter=" ", dtype=np.float32)
    )
    high_res_signals = torch.from_numpy(
        np.loadtxt(f"{train_data_dir}/high_res.txt", delimiter=" ", dtype=np.float32)
    )

    return low_res_signals, high_res_signals


def main():
    args = parse_args()
    model_name = args.model_name
    low_res_signals, high_res_signals = get_datasets(model_name)

    client = Client('127.0.0.1:8786')

    net = NeuralNet(
        module=ActTime,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        max_epochs=1,
        callbacks=[training.EarlyStopping(patience=30)],
        batch_size=64,
        verbose=1
    )

    params = {
        "module__n_channels": [1],
        "module__n_feats": [64, 128],
        "module__token_size": [3],
        "module__n_heads": [8, 16],
        "module__n_layers": [8],
        "module__scale": [5],
        "module__reduction": [16],
        "module__n_resblocks": [12],
        "module__n_resgroups": [4],
        "module__n_fusionblocks": [4],
        "module__expansion_ratio": [4],
        "lr": uniform(loc=0.0001, scale=0.0009),
    }
    gs = RandomizedSearchCV(
        estimator=net,
        param_distributions=params,
        scoring="neg_mean_squared_error",
        cv=2,
        verbose=3,
        random_state=0,
        error_score='raise',
        n_iter=1,
        n_jobs=-1
    )

    with parallel_backend('dask'):
        gs.fit(low_res_signals, high_res_signals)
        
    # make directory to save experiment
    save_dir = 'experiments/train'
    if not args.save_path:
        now = datetime.datetime.now().strftime('%Y-%m-%d')
        args.save_path = os.path.join(save_dir, now) 
    else: 
        args.save_path = os.path.join(save_dir, args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    print(f'Experimental results will be saved at: {args.save_path}')
    # print(gs.best_score_, gs.best_params_)
    # gs.best_estimator_.save_params(f_params=f"saved_models/model_{model_name}.pkl")

    gs.best_estimator_.save_params(f_params=f"{args.save_path}/{model_name}.pkl")
    with open(f'{args.save_path}/{model_name}.json', 'w') as fp:
        json.dump(gs.best_params_, fp)
        
    # net = NeuralNet(
    #     module=ActTime,
    #     criterion=nn.MSELoss,
    #     max_epochs=1,
    #     batch_size=64,
    #     optimizer=torch.optim.Adam,
    #     lr=0.001,
    #     module__args=args,
    #     callbacks=[training.EarlyStopping(patience=30)],
    #     # verbose=0,
    # )

    # net.fit(low_res_signals, high_res_signals)
    # net.save_params(f_params=f"saved_models/model_{model_name}.pkl")

if __name__ == "__main__":
    main()
