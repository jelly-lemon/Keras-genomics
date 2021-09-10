import argparse, numpy as np, h5py
from os.path import exists, dirname, join, basename
from os import makedirs

# 映射规则
MAPPER = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}


def arr2HDF5file(data, label, out_file_path, label_name, data_name) -> None:
    """
    array 转 HDF5 文件，文件后缀为 .batch（将数据和标签保存在同一个 batch file 中）

    :param data:
    :param label:
    :param out_file_path: 保存文件路径
    :param label_name:
    :param data_name:
    :return:
    """
    print('batch_files shape: ', data.shape)
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    # label = [[x.astype(np.float32)] for x in label]
    with h5py.File(out_file_path, 'w') as file:
        file.create_dataset(name=data_name, data=data, **comp_kwargs)
        file.create_dataset(name=label_name, data=label, **comp_kwargs)


def seq2feature(data, mapper, label, out_filename, word_dim, label_name, data_name):
    """
    序列()转 HDF5 文件

    :param data: 数据
    :param mapper: 数据映射规则
    :param label: 标签
    :param out_filename: 输出文件名
    :param word_dim:
    :param label_name:
    :param data_name:
    """
    # 读取每条序列，映射为数字矩阵
    out = []
    for seq in data:
        mat = seq2onehot(seq, mapper, word_dim)
        result = mat.transpose()
        result1 = [[a] for a in result]
        out.append(result1)

    # 将矩阵保存为 hdf5 文件
    arr2HDF5file(np.asarray(out), label, out_filename, label_name, data_name)


def feature2feature(data, mapper, label, out_filename, worddim, labelname, dataname):
    """

    :param data:
    :param mapper:
    :param label:
    :param out_filename:
    :param worddim:
    :param labelname:
    :param dataname:
    :return:
    """
    out = np.asarray(data)[:, None, None, :]
    arr2HDF5file(out, label, out_filename, labelname, dataname)


def seq2onehot(seq, word_dim):
    """
    序列(ATCG)转 one-hot 矩阵

    :param seq: 序列
    :param word_dim: 词维度（即一个数据的长度）
    :return: one-hot 矩阵
    """
    mat = np.asarray([MAPPER[element] if element in MAPPER else np.random.rand(word_dim) * 2 - 1 for element in seq])
    return mat


def seq2feature_siamese(data1, data2, mapper, label, out_filename, worddim, labelname, dataname):
    out = []
    datalen = len(data1)
    for dataidx in range(datalen):
        mat = np.asarray([seq2onehot(data1[dataidx], mapper, worddim), seq2onehot(data2[dataidx], mapper, worddim)])
        result = mat.transpose((2, 0, 1))
        out.append(result)
    arr2HDF5file(np.asarray(out), label, out_filename, labelname, dataname)


def tsv2batchfile(infile, label_file_path, out_file_path, mapper, worddim, batch_size, labelname, dataname, isseq):
    """
    对输入文件进行格式转换，tsv -> batch

    :param infile:
    :param label_file_path:
    :param out_file_path:
    :param mapper:
    :param worddim:
    :param batch_size:
    :param labelname:
    :param dataname:
    :param isseq:
    :return:批数量
    """
    with open(infile) as seq_file, open(label_file_path) as label_file:
        cnt = 0
        seq_data = []
        label = []
        batch_num = 0

        # 将序列和标签合并在一起
        for x, y in zip(seq_file, label_file):
            if isseq:
                seq_data.append(list(x.strip().split()[1]))
            else:
                seq_data.append(map(float, x.strip().split()))
            label.append(list(map(float, y.strip().split())))

            # 满一个 batch 时，转为 feature，并存储为文件
            cnt = (cnt + 1) % batch_size
            if cnt == 0:
                batch_num = batch_num + 1
                seq_data = np.asarray(seq_data)
                label = np.asarray(label)
                t_outfile = out_file_path + '.batch' + str(batch_num)
                if isseq:
                    seq2feature(seq_data, mapper, label, t_outfile, worddim, labelname, dataname)
                else:
                    feature2feature(seq_data, mapper, label, t_outfile, worddim, labelname, dataname)
                seq_data = []
                label = []

        # 多余样本
        if cnt > 0:
            batch_num = batch_num + 1
            seq_data = np.asarray(seq_data)
            label = np.asarray(label)
            t_outfile = out_file_path + '.batch' + str(batch_num)
            if isseq:
                seq2feature(seq_data, mapper, label, t_outfile, worddim, labelname, dataname)
            else:
                feature2feature(seq_data, mapper, label, t_outfile, worddim, labelname, dataname)
    return batch_num


def convert_siamese(infile1, infile2, labelfile, outfile, mapper, worddim, batchsize, labelname, dataname):
    with open(infile1) as seqfile1, open(infile2) as seqfile2, open(labelfile) as labelfile:
        cnt = 0
        seqdata1 = []
        seqdata2 = []
        label = []
        batchnum = 0
        for x1, x2, y in zip(seqfile1, seqfile2, labelfile):
            seqdata1.append(list(x1.strip().split()[1]))
            seqdata2.append(list(x2.strip().split()[1]))
            # label.append(float(y.strip()))
            label.append(map(float, y.strip().split()))
            cnt = (cnt + 1) % batchsize
            if cnt == 0:
                batchnum = batchnum + 1
                seqdata1 = np.asarray(seqdata1)
                seqdata2 = np.asarray(seqdata2)
                label = np.asarray(label)
                t_outfile = outfile + '.batch' + str(batchnum)
                seq2feature_siamese(seqdata1, seqdata2, mapper, label, t_outfile, worddim, labelname, dataname)
                seqdata1 = []
                seqdata2 = []
                label = []

        if cnt > 0:
            batchnum = batchnum + 1
            seqdata1 = np.asarray(seqdata1)
            seqdata2 = np.asarray(seqdata2)
            label = np.asarray(label)
            t_outfile = outfile + '.batch' + str(batchnum)
            seq2feature_siamese(seqdata1, seqdata2, mapper, label, t_outfile, worddim, labelname, dataname)

    return batchnum


def manifest(out_filename, batch_num, prefix):
    """
    保存清单文件（txt 格式，里面说明了 batch_file 有多少个）

    :param out_filename: 输出文件名
    :param batch_num: 批文件数量
    :param prefix: 每行输出前缀
    """
    out_file_path = join(dirname(out_filename), basename(out_filename).split('.')[0] + '.txt')
    with open(out_file_path, 'w') as f:
        for i in range(batch_num):
            f.write('.'.join(['/'.join([prefix] + out_filename.split('/')[-2:]), 'batch' + str(i + 1)]) + '\n')


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Convert sequence and target for Caffe")

    # 位置参数（必须输入）
    parser.add_argument("infile", type=str, help="Sequence in FASTA/TSV format (with .fa/.fasta or .tsv extension)")
    parser.add_argument("labelfile", type=str, help="Label of the sequence. One number per line")
    parser.add_argument("outfile", type=str, help="Output file (example: $MODEL_TOPDIR$/batch_files/train.h5). ")

    # 可选选项
    parser.add_argument("-m", "--mapperfile", dest="mapperfile", default="",
                        help="A TSV file mapping each nucleotide to a vector. The first column should be the nucleotide, and the rest denote the vectors. (Default mapping: A:[1,0,0,0],C:[0,1,0,0],G:[0,0,1,0],T:[0,0,0,1])")
    # 孪生神经网络输入文件
    parser.add_argument("-i", "--infile2", dest="infile2", default="", help="The paired input file for siamese network")
    parser.add_argument("-b", "--batch", dest="batch", type=int, default=5000,
                        help="Batch size for batch_files storage (Defalt:5000)")
    parser.add_argument("-p", "--prefix", dest="maniprefix", default='/batch_files',
                        help="The model_dir (Default: /batch_files . This only works for mri-wrapper)")
    parser.add_argument("-l", "--labelname", dest="labelname", default='label',
                        help="The group name for labels in the HDF5 file")
    parser.add_argument("-d", "--dataname", dest="dataname", default='batch_files',
                        help="The group name for batch_files in the HDF5 file")
    parser.add_argument("-s", "--isseq", dest="isseq", default='Y',
                        help="The group name for batch_files in the HDF5 file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 获取输出目录，不存在则创建
    out_dir = dirname(args.outfile)
    if not exists(out_dir):
        makedirs(out_dir)

    # 读取输入文件，并进行格式转换
    if args.infile2 == '':
        print(args.isseq == 'Y')
        batch_num = tsv2batchfile(args.infile, args.labelfile, args.outfile, MAPPER, len(MAPPER['A']),
                                  args.batch,
                                  args.labelname, args.dataname, args.isseq == 'Y')
    else:
        batch_num = convert_siamese(args.infile, args.infile2, args.labelfile, args.outfile, MAPPER,
                                    len(MAPPER['A']), args.batch, args.labelname, args.dataname)

    # 保存到文件
    manifest(args.outfile, batch_num, args.maniprefix)
