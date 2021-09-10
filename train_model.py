from __future__ import print_function

import argparse
import os
import DataHelper
from models.ModelHelper import *

# 当前工作目录
cwd = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("-m", "--model_name", dest="model_name", help="Model name")
    parser.add_argument("-r", "--retrain", dest="retrain", default=False, action='store_true', help="whether retrain model")

    return parser.parse_args()








if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    print(args)
    output_dir = "./saved_models"
    data_dir = "./batch_files"

    # 加载模型
    if args.model_name is None:
        raise Exception("please input model name")
    baseModel = get_model(args.model_name, output_dir, args.retrain)

    # 加载数据
    x_train, y_train = DataHelper.read_data(data_dir + "/train.h5.batch")
    x_val, y_val = DataHelper.read_data(data_dir + "/valid.h5.batch")

    #
    # 训练
    #
    baseModel.train(x_train, y_train, x_val, y_val)

"""
    elif args.retrain:
        #
        # 加载权重继续训练
        # -d batch_files -m EasyModel.py -te 1000  -pa 100 -r r01 -rw batch_files/model/model_bestmodel_weights.h5
        #
        ### Resume training
        new_weight_file = weight_file_out_path + '.' + args.retrain
        new_last_weight_file = last_weight_file + '.' + args.retrain

        model = baseModel.load_saved_model()  # 加载模型

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
