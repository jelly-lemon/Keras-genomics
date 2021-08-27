from __future__ import print_function

import argparse
import h5py
import os
# python2 中叫 cPickle
import pickle as cPickle
import shutil
import sys
import time
from pprint import pprint
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras import Model
import MyModel

import DataHelper
from hyperband import Hyperband

# 当前工作目录
cwd = os.path.dirname(os.path.realpath(__file__))

#
# 默认超参数
#
LOSS = "binary_crossentropy"
OPTIMIZER = Adam()
METRICS = "[acc]"


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("-y", "--hyper", dest="hyper", default=False, action='store_true',
                        help="Perform hyper-parameter tuning")
    parser.add_argument("-t", "--train", dest="train", default=False, action='store_true',
                        help="Train on the training set with the best hyper-params")
    parser.add_argument("-e", "--eval", dest="eval", default=False, action='store_true',
                        help="Evaluate the model on the test set")
    parser.add_argument("-p", "--predit", dest="infile", default='',
                        help="Path to batch_file to predict on (up till batch number)")
    parser.add_argument("-d", "--topdir", dest="topdir", help="The batch_file directory")
    parser.add_argument("-m", "--model", dest="model", help="Path to the model file")
    parser.add_argument("-o", "--outdir", dest="outdir", default='',
                        help="Output directory for the prediction on new batch_file")
    parser.add_argument("-hi", "--hyperiter", dest="hyperiter", default=20, type=int,
                        help="Num of max iteration for each hyper-param config")
    parser.add_argument("-te", "--trainepoch", default=20, type=int, help="The number of epochs to train for")
    parser.add_argument("-pa", "--patience", default=10, type=int,
                        help="number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("-bs", "--batchsize", default=100, type=int, help="Batchsize in SGD-based training")
    parser.add_argument("-w", "--weightfile", default=None, help="Weight file for the best model")
    parser.add_argument("-l", "--lweightfile", default=None, help="Weight file after training")
    parser.add_argument("-r", "--retrain", default=None, help="codename for the retrain run")
    parser.add_argument("-rw", "--rweightfile", default='', help="Weight file to load for retraining")
    parser.add_argument("-dm", "--datamode", default='memory',
                        help="whether to load batch_file into memory ('memory') or using a generator('generator')")
    parser.add_argument("-ei", "--evalidx", dest='evalidx', default=0, type=int,
                        help="which output neuron (0-based) to calculate 2-class auROC for")
    parser.add_argument("--epochratio", default=1, type=float,
                        help="when training with batch_file generator, optionally shrink each epoch size by this factor to enable more frequen evaluation on the valid set")
    parser.add_argument("-shuf", default=1, type=int,
                        help="whether to shuffle the batch_file at the begining of each epoch (1/0)")

    return parser.parse_args()


def train_model(model: Model, weight_save_path: str):
    """
    训练模型

    :param model:模型
    :param weight_save_path: 权重保存路径
    """
    checkpointer = ModelCheckpoint(filepath=weight_save_path, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=0)

    # 加载数据、设置超参数
    if args.datamode == 'generator':
        trainbatch_num, train_size = DataHelper.probe_data(os.path.join(args.topdir, 'train.h5.batch'))
        validbatch_num, valid_size = DataHelper.probe_data(os.path.join(args.topdir, 'valid.h5.batch'))
        history_callback = model.fit_generator(
            DataHelper.batch_generator(args.batchsize, os.path.join(args.topdir, 'train.h5.batch'), shuf=args.shuf == 1),
            train_size / args.batchsize * args.epochratio,
            args.trainepoch,
            validation_data=DataHelper.batch_generator(args.batchsize, os.path.join(args.topdir, 'valid.h5.batch'),
                                               shuf=args.shuf == 1),
            validation_steps=np.ceil(float(valid_size) / args.batchsize),
            callbacks=[checkpointer, early_stopping])
    else:
        Y_train, traindata = DataHelper.read_data(os.path.join(args.topdir, 'train.h5.batch'))
        Y_valid, validdata = DataHelper.read_data(os.path.join(args.topdir, 'valid.h5.batch'))
        history_callback = model.fit(
            traindata,
            Y_train,
            batch_size=args.batchsize,
            epochs=args.trainepoch,
            validation_data=(validdata, Y_valid),
            callbacks=[checkpointer, early_stopping],
            shuffle=args.shuf == 1)

    return model, history_callback


def load_model(architecture_file:str = None, weightfile2load: str = None, optimizer_file:str=None) -> Model:
    """
    加载模型

    :param weightfile2load:权重文件路径 [in]
    :return:模型
    """
    # 加载模型结构
    try:
        model = model_from_json(open(architecture_file).read())
        print("model_from_json succeed")
    except Exception as e:
        # TODO
        model = MyModel.get_model()
        print("get_model succeed")


    # 加载保存的权重
    if weightfile2load:
        model.load_weights(weightfile2load)

    # 配置超参数
    try:
        best_optimizer, best_optimizer_config, best_loss = cPickle.load(open(optimizer_file, 'rb'))
        #TODO 根据 best_optimizer_config 设置 best_optimizer
        model.compile(loss=best_loss, optimizer=best_optimizer, metrics=METRICS)
    except Exception:
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

    return model


if __name__ == "__main__":
    os.system("chcp 65001")

    # 解析命令行参数
    args = parse_args()

    # 获取模型文件名
    model_arch = os.path.basename(args.model)
    model_arch = model_arch[:-3] if model_arch[-3:] == '.py' else model_arch

    # 创建输出目录
    outdir = os.path.join(args.topdir, model_arch)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 设置模型结构、最优超参数、权重、上一次权重保存路径
    architecture_file = os.path.join(outdir, model_arch + '_best_archit.json')
    optimizer_file = os.path.join(outdir, model_arch + '_best_optimer.pkl')
    weight_file_out_path = os.path.join(outdir,
                                        model_arch + '_bestmodel_weights.h5') if args.weightfile is None else args.weightfile
    last_weight_file = os.path.join(outdir,
                                    model_arch + '_lastmodel_weights.h5') if args.lweightfile is None else args.lweightfile
    evalout = os.path.join(outdir, model_arch + '_eval.txt')

    #
    # 把模型放在临时目录，这样不管传什么模型进行，统一重命名为 mymodel.py，复用代码
    #
    tmpdir = "./tmp"
    if os.path.exists(tmpdir) is False:
        os.makedirs(tmpdir)
    shutil.copy(args.model, './tmp/mymodel.py')
    sys.path.append(tmpdir)

    #
    # 超参数搜索
    #
    if args.hyper:
        # 搜索最优参数
        hb = Hyperband(MyModel.get_params, MyModel.try_params, args.topdir, max_iter=args.hyperiter,
                       datamode=args.datamode)
        results = hb.run(skip_last=1)

        # 打印最优参数
        best_result = sorted(results, key=lambda x: x['loss'])[0]
        pprint(best_result['params'])

        # 保存最优参数到文件 architecture_file
        best_archit, best_optim, best_optim_config, best_lossfunc = best_result['model']
        open(architecture_file, 'w').write(best_archit)
        cPickle.dump((best_optim, best_optim_config, best_lossfunc), open(optimizer_file, 'wb'))

    #
    # 训练
    #
    if args.train:
        model = load_model()
        model, history_callback = train_model(model, weight_file_out_path)
        model.save_weights(last_weight_file, overwrite=True)
        myhist = history_callback.history
        all_hist = np.asarray([myhist["loss"], myhist["acc"], myhist["val_loss"],
                               myhist["val_acc"]]).transpose()
        np.savetxt(os.path.join(outdir, model_arch + ".training_history.txt"), all_hist, delimiter="\t",
                   header='loss\tacc\tval_loss\tval_acc')
    elif args.retrain:
        #
        # 加载权重继续训练
        # -d batch_file -m MyModel.py -te 1000  -pa 100 -r r01 -rw batch_file/model/model_bestmodel_weights.h5
        #
        ### Resume training
        new_weight_file = weight_file_out_path + '.' + args.retrain
        new_last_weight_file = last_weight_file + '.' + args.retrain

        model = load_model(args.rweightfile)
        model, history_callback = train_model(model, new_weight_file)

        model.save_weights(new_last_weight_file, overwrite=True)
        myhist = history_callback.history
        all_hist = np.asarray([myhist["loss"], myhist["categorical_accuracy"], myhist["val_loss"],
                               myhist["val_categorical_accuracy"]]).transpose()
        np.savetxt(os.path.join(outdir, model_arch + ".training_history." + args.retrain + ".txt"), all_hist,
                   delimiter="\t",
                   header='loss\tacc\tval_loss\tval_acc')

    #
    # 评估
    #
    if args.eval:
        ## Evaluate
        model = load_model(weight_file_out_path)

        pred_for_evalidx = []
        pred_bin = []
        y_true_for_evalidx = []
        y_true = []
        testbatch_num, _ = hb.probe_data(os.path.join(args.topdir, 'test.h5.batch'))
        test_generator = hb.batch_generator(None, os.path.join(args.topdir, 'test.h5.batch'), shuf=args.shuf == 1)
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
        np.savetxt(evalout, [t_auc, t_acc])

    #
    # 预测未知数据
    #
    if args.infile != '':
        ## Predict on new batch_file
        model = load_model(weight_file_out_path)

        predict_batch_num, _ = hb.probe_data(args.infile)
        print('Total number of batch to predict:', predict_batch_num)

        outdir = os.path.join(os.path.dirname(args.infile),
                              '.'.os.path.join(['pred', model_arch,
                                                os.path.basename(args.infile)])) if args.outdir == '' else args.outdir
        if os.path.exists(outdir):
            print('Output directory', outdir, 'exists! Overwrite? (yes/no)')
            if input().lower() == 'yes':
                shutil.rmtree(outdir)
            else:
                print('Quit predicting!')
                sys.exit(1)

        for i in range(predict_batch_num):
            print('predict on batch', i)
            batch_data = h5py.File(args.infile + str(i + 1), 'r')['batch_file']

            time1 = time.time()
            pred = model.predict(batch_data)
            time2 = time.time()
            print('predict took %0.3f ms' % ((time2 - time1) * 1000.0))

            t_outdir = os.path.join(outdir, 'batch' + str(i + 1))
            os.makedirs(t_outdir)
            for label_dim in range(pred.shape[1]):
                with open(os.path.join(t_outdir, str(label_dim) + '.pkl'), 'wb') as f:
                    cPickle.dump(pred[:, label_dim], f)

    # 删除临时文件
    shutil.rmtree(tmpdir)
