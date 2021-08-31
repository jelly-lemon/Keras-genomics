from __future__ import print_function

import argparse
import h5py
import os
import pickle
import shutil
import sys
import time
from pprint import pprint
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

import DataHelper
from Hyperband import Hyperband
from models.Util import *
import json

# 当前工作目录
cwd = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("--save_tag", dest="save_tag", default="", action='store_true', help="Output directory tag")
    parser.add_argument("--hyper", dest="hyper", default=False, action='store_true',
                        help="Perform hyper-parameter tuning")
    parser.add_argument("-t", "--train", dest="train", default=False, action='store_true',
                        help="Train on the training set with the best hyper-params")
    parser.add_argument("-e", "--eval", dest="eval", default=False, action='store_true',
                        help="Evaluate the model on the test set")
    parser.add_argument("-p", "--predit", dest="infile", default='',
                        help="Path to batch_files to predict on (up till batch number)")
    parser.add_argument("-d", "--data_dir", dest="data_dir", default='./batch_files',
                        help="The batch_files directory")
    parser.add_argument("-m", "--model_name", dest="model_name", help="Model name")
    parser.add_argument("-o", "--out_dir", dest="out_dir", default='./saved_models',
                        help="Model output directory")
    parser.add_argument("-te", "--trainepoch", default=20, type=int, help="The number of epochs to train for")
    parser.add_argument("-pa", "--patience", default=10, type=int,
                        help="number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("-bs", "--batchsize", default=100, type=int, help="Batchsize in SGD-based training")
    parser.add_argument("-w", "--weightfile", default=None, help="Weight file for the best model")
    parser.add_argument("-l", "--lweightfile", default=None, help="Weight file after training")
    parser.add_argument("-r", "--retrain", default=None, help="codename for the retrain run")
    parser.add_argument("-rw", "--rweightfile", default='', help="Weight file to load for retraining")
    parser.add_argument("-dm", "--data_mode", default='memory',
                        help="whether to load batch_files into memory ('memory') or using a generator('generator')")
    parser.add_argument("-ei", "--evalidx", dest='evalidx', default=0, type=int,
                        help="which output neuron (0-based) to calculate 2-class auROC for")
    parser.add_argument("--epochratio", default=1, type=float,
                        help="when training with batch_files generator, optionally shrink each epoch size by this factor to enable more frequen evaluation on the valid set")
    parser.add_argument("-shuf", default=1, type=int,
                        help="whether to shuffle the batch_files at the begining of each epoch (1/0)")

    return parser.parse_args()








if __name__ == "__main__":
    os.system("chcp 65001")

    # 解析命令行参数
    args = parse_args()

    # 获取模型名
    model_name = args.model_name
    if model_name is None:
        raise Exception("please input model name")

    # 加载模型
    baseModel = get_model(model_name, args.out_dir, args.save_tag)

    # 获取输出目录
    out_dir = baseModel.save_dir

    # 设置模型结构等保存路径
    architecture_file = os.path.join(out_dir, model_name + '_best_archit.json')
    optimizer_file = os.path.join(out_dir, model_name + '_best_optimer.json')
    weight_file_out_path = os.path.join(out_dir,
                                        model_name + '_bestmodel_weights.h5') if args.weightfile is None else args.weightfile
    last_weight_file = os.path.join(out_dir,
                                    model_name + '_lastmodel_weights.h5') if args.lweightfile is None else args.lweightfile
    eval_out_path = os.path.join(out_dir, model_name + '_eval.txt')

    # 加载数据
    print(args)
    x_train, y_train = DataHelper.read_data(args.data_dir + "/train.h5.batch")
    x_valid, y_valid = DataHelper.read_data(args.data_dir + "/valid.h5.batch")




    #
    # 超参数搜索
    #
    if args.hyper:
        # 搜索最优参数
        hb = Hyperband(baseModel)
        results = hb.search(x_train, y_train, x_valid, y_valid,
                            skip_last=1)

        # 保存最优参数到文件 architecture_file
        best_result = sorted(results, key=lambda x: x['val_loss'])[0]
        print(best_result)
        json.dump(best_result['model'], open(architecture_file, 'w'))
        json.dump(best_result["params"], open(optimizer_file, 'w'))

    """
    #
    # 训练
    # -d batch_files -m ./models/EasyModel.py -te 1000  -pa 100 -t
    #
    if args.train:
        model = load_saved_model()  # 加载模型
        model, history_callback = train_model(model, weight_file_out_path)  # 训练模型
        model.save_weights(last_weight_file, overwrite=True)  # 保存权重，覆盖旧权重
        # 保存训练数据到文件
        myhist = history_callback.history
        all_hist = np.asarray([myhist["loss"], myhist["acc"], myhist["val_loss"],
                               myhist["val_acc"]]).transpose()
        np.savetxt(os.path.join(out_dir, model_name + ".training_history.txt"), all_hist, delimiter="\t",
                   header='loss\tacc\tval_loss\tval_acc')

    elif args.retrain:
        #
        # 加载权重继续训练
        # -d batch_files -m EasyModel.py -te 1000  -pa 100 -r r01 -rw batch_files/model/model_bestmodel_weights.h5
        #
        ### Resume training
        new_weight_file = weight_file_out_path + '.' + args.retrain
        new_last_weight_file = last_weight_file + '.' + args.retrain

        model = load_saved_model(args.rweightfile)
        model, history_callback = train_model(model, new_weight_file)

        model.save_weights(new_last_weight_file, overwrite=True)
        myhist = history_callback.history
        all_hist = np.asarray([myhist["loss"], myhist["categorical_accuracy"], myhist["val_loss"],
                               myhist["val_categorical_accuracy"]]).transpose()
        np.savetxt(os.path.join(out_dir, model_name + ".training_history." + args.retrain + ".txt"), all_hist,
                   delimiter="\t",
                   header='loss\tacc\tval_loss\tval_acc')

    #
    # 评估
    #
    if args.eval:
        ## Evaluate
        model = load_saved_model(weight_file_out_path)

        pred_for_evalidx = []
        pred_bin = []
        y_true_for_evalidx = []
        y_true = []
        testbatch_num, _ = hb.probe_data(os.path.join(args.data_dir, 'test.h5.batch'))
        test_generator = hb.batch_generator(None, os.path.join(args.data_dir, 'test.h5.batch'), shuf=args.shuf == 1)
        for _ in range(testbatch_num):
            X_test, Y_test = next(test_generator)
            t_pred = model.predict(X_test)
            pred_for_evalidx += [x[args.evalidx] for x in t_pred]
            pred_bin += [np.argmax(x) for x in t_pred]
            y_true += [np.argmax(x) for x in Y_test]
            y_true_for_evalidx += [x[args.evalidx] for x in Y_test]

        t_auc = roc_auc_score(y_true_for_evalidx, pred_for_evalidx)
        t_acc = accuracy_score(y_true, pred_bin)
        print('Test AUC for output neuron {}:'.format(args.evalidx), t_auc)
        print('Test categorical accuracy:', t_acc)
        np.savetxt(eval_out_path, [t_auc, t_acc])

    #
    # 预测未知数据
    #
    if args.infile != '':
        ## Predict on new batch_files
        model = load_saved_model(weight_file_out_path)

        predict_batch_num, _ = hb.probe_data(args.infile)
        print('Total number of batch to predict:', predict_batch_num)

        out_dir = os.path.join(os.path.dirname(args.infile),
                               '.'.os.path.join(['pred', model_name,
                                                 os.path.basename(args.infile)])) if args.outdir == '' else args.outdir
        if os.path.exists(out_dir):
            print('Output directory', out_dir, 'exists! Overwrite? (yes/no)')
            if input().lower() == 'yes':
                shutil.rmtree(out_dir)
            else:
                print('Quit predicting!')
                sys.exit(1)

        for i in range(predict_batch_num):
            print('predict on batch', i)
            batch_data = h5py.File(args.infile + str(i + 1), 'r')['batch_files']

            time1 = time.time()
            pred = model.predict(batch_data)
            time2 = time.time()
            print('predict took %0.3f ms' % ((time2 - time1) * 1000.0))

            t_outdir = os.path.join(out_dir, 'batch' + str(i + 1))
            os.makedirs(t_outdir)
            for label_dim in range(pred.shape[1]):
                with open(os.path.join(t_outdir, str(label_dim) + '.pkl'), 'wb') as f:
                    pickle.dump(pred[:, label_dim], f)

"""
