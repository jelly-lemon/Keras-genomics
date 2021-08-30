import os

import numpy as np
from os.path import join
from random import random
from math import log, ceil
from time import time, ctime
import DataHelper
from models.BaseModel import BaseModel


class Hyperband:
    def __init__(self, baseModel:BaseModel, data_dir, eta=3, data_mode='memory'):
        self.baseModel = baseModel
        self.max_epochs = 20

        # 读取数据到内存
        if data_mode == 'memory':
            Y_train, X_train = DataHelper.read_data(data_dir + "/" + 'train.h5.batch')
            Y_test, X_test = DataHelper.read_data(data_dir + "/" + 'valid.h5.batch')
            self.data = {'train': (X_train, Y_train), 'valid': (X_test, Y_test)}
        else:
            # 数据生成器
            self.data = {
                'train': {
                    'gen_func': DataHelper.batch_generator,
                    'path': data_dir + "/" + 'train.h5.batch',
                    'n_sample': DataHelper.probe_data(join(data_dir, 'train.h5.batch'))[1]},
                'valid': {
                    'gen_func': DataHelper.batch_generator,
                    'path': data_dir + "/" + 'valid.h5.batch',
                    'n_sample': DataHelper.probe_data(join(data_dir, 'valid.h5.batch'))[1]},
            }

        self.data_mode = data_mode
        self.max_iter = self.max_epochs  # maximum iterations per configuration
        self.eta = eta  # defines configuration downsampling rate (default = 3)

        self.log_eta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.log_eta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    def run(self, skip_last=0, dry_run=False):
        """

        :param skip_last:
        :param dry_run:
        :return:
        """

        for s in reversed(range(self.s_max + 1)):
            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # 生成 n 种随机超参数配置
            T_params = [self.baseModel.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                epochs = int(r * self.eta ** (i))

                # print("\n*** {} configurations x {:.1f} iterations each".format(
                #     n_configs, n_iterations))

                val_losses = []
                early_stops = []

                # 尝试所有配置
                for params in T_params:

                    self.counter += 1
                    print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
                        self.counter, ctime(), self.best_loss, self.best_counter))

                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.baseModel.try_params(epochs, params, self.data, self.data_mode)  # <---

                    assert (type(result) == dict)
                    assert ('loss' in result)

                    seconds = int(round(time() - start_time))
                    print("\n{} seconds.".format(seconds))

                    loss = result['loss']
                    val_losses.append(loss)

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = params
                    result['epochs'] = epochs

                    self.results.append(result)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T_params = [T_params[i] for i in indices if not early_stops[i]]
                T_params = T_params[0:int(n_configs / self.eta)]

        return self.results






