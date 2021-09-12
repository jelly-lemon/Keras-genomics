import argparse, numpy as np, h5py
from os.path import exists, dirname, join, basename
from os import makedirs

# 映射规则
MAPPER = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
DEFAULT_WORD_DIM = len(MAPPER['A'])  # 即每个字符的对应的 one-hot 长度


def arr2h5(data: np.ndarray, label: np.ndarray, out_file_path: str,
           data_name: str, label_name: str) -> None:
    """
    array 转 HDF5 文件，文件后缀为 .batch（将数据和标签保存在同一个 batch file 中）

    :param data: 数据
    :param label: 标签
    :param out_file_path: 保存文件路径
    :param data_name: data 保存在 h5 文件中的 dataset 名
    :param label_name: label 的 dataset 保存名
    """
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(out_file_path, 'w') as file:
        file.create_dataset(name=data_name, data=data, **comp_kwargs)
        file.create_dataset(name=label_name, data=label, **comp_kwargs)


def seq2feature(sequences: np.ndarray) -> np.ndarray:
    """
    序列转 feature

    :param sequences: 序列
    :return feature
    """
    # 读取每条序列，映射为数字矩阵
    features = []
    for seq in sequences:
        mat = seq2onehot(seq)
        result = mat.transpose()
        result1 = [[a] for a in result]
        features.append(result1)

    features = np.asarray(features)
    return features


def seq2onehot(seq: str) -> np.ndarray:
    """
    序列(ATCG)转 one-hot 矩阵

    :param seq: 序列
    :return: one-hot 矩阵
    """
    mat = [MAPPER[element] if element in seq else MAPPER["N"] for element in seq]
    mat = np.asarray(mat)

    return mat


def tsv2h5(seq_file_path: str, label_file_path: str, out_dir: str,
           batch_file_size: int = 32 * 200) -> int:
    """
    读取 tsv 并转存为 h5 文件

    整体流程为：
    *.tsv -> 读取序列（如："ATCG..."） -> 转为 feature (如：[[0 1 0 0], ...]) -> 保存为 h5 文件

    :param seq_file_path: 数据路径
    :param label_file_path: 标签路径
    :param out_dir: 输出目录
    :param batch_file_size: 批文件容量，即生成的 h5 文件存放的数据量
    :return 生成的文件数
    """
    # 获取数据文件前缀
    seq_file_prefix = basename(seq_file_path).split('.')[0]

    # 转换
    with open(seq_file_path) as seq_file, open(label_file_path) as label_file:
        count = 0
        sequences = []
        labels = []
        batch_num = 0

        # 将序列和标签合并在一起
        # zip 两个文件，会按行返回
        for i, (seq, label) in enumerate(zip(seq_file, label_file)):
            sequences.append(list(seq.strip().split()[1]))
            labels.append(list(map(float, label.strip().split())))

            # 满一个 batch 时，转为 feature，并存储为文件
            count = (count + 1) % batch_file_size
            if count == 0:
                batch_num = batch_num + 1
                new_out_file_path = out_dir + "/" + seq_file_prefix + '.h5.batch' + str(batch_num)
                sequences = np.asarray(sequences)
                labels = np.asarray(labels)
                features = seq2feature(sequences)

                # 保存到文件
                arr2h5(features, labels, new_out_file_path, "data" + str(batch_num), "label" + str(batch_num))

                sequences = []
                labels = []

        # 多余样本（不足一个 batch，直接抛弃）

    return batch_num

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="tsv2h5")

    # Positional (unnamed) arguments:
    parser.add_argument("tsv_file",  type=str, help="tsf file path")
    parser.add_argument("label_file",  type=str,help="label file path")
    parser.add_argument("out_dir",  type=str, help="output directory")

    return parser.parse_args()

if __name__ == "__main__":
    # 获取输出目录，不存在则创建
    out_dir = "../row_data/batch_files"
    if not exists(out_dir):
        makedirs(out_dir)

    # 读取输入文件，并进行格式转换
    tsv2h5("../row_data/test.tsv", "../row_data/test.target", out_dir)

