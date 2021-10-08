基于 tensorflow/keras 对基因组数据（genomics data）进行深度学习训练、预测和超参数调整。

fork 自：https://github.com/gifford-lab/Keras-genomics

预测是否存在 motif

主页：http://cnn.csail.mit.edu/

论文：Zeng H., Edwards M.D., Gifford D. K.(2015) "Convolutional Neural Network Architectures for Predicting DNA-Protein Binding".
Proceedings of Intelligent Systems for Molecular Biology (ISMB) 2016
Bioinformatics, 32(12):i121-i127. doi: 10.1093/bioinformatics/btw255.

论文作者 GitHub 源码：https://github.com/gifford-lab/Keras-genomics

简介：数据集包含大量 DNA结合蛋白序列 片段，如："CAGTTGGCC...CAAAGGGAACACACAAGTAGA"，以及对应标签 1。
标签 1 代表该片段上有结合位点（称之为 motif，具体来看就是一个子串），标签 0 代表不存在。所以，这是一个
二分类任务。

# 数据集

数据集下载：http://cnn.csail.mit.edu/motif_discovery/

序列长这样：
```
>chr5:114910510-114910610 		GCCACCACGCCTGACTAATTTTTGTATTTTCAGTAGAGATGGGGTTTCACCGTCTTGGCCAGGCTGGTCTTGAACTACTGACCTCGTGATCCACCCACCTT 1
>chr5:114910510-114910610_shuf 	GACCTTGTTTACACCTTGGCACCAATCACTGATGCAACCTATTCGGTTGACTTCCTAGTTGACTCCAGGGGCTTTTACGTGATCCCCAGTTCCGGCTGAGT 0
```
正样本： 以 Chip-seq peak 为中心的 101 个 bp 序列。

negative set:

# 运行环境
win 10

python 3.6

## 相关 python 包（必须一致）
tensorflow 2.1.4（不用再单独安装 Keras，tf 2 自带 Keras）

hyperopt 0.2.5	选择超参数

**务必保证所有版本一致，避免出现未知的错误！**

## 运行方式
1. 安装 Anaconda

2. 创建基于 Python 3.6 的虚拟环境

3. 安装 PyCharm（版本随意）

4. 克隆本项目到本地

5. 安装相关 Python 依赖包

6. 运行

## 文件夹说明
--Keras-genomics
  |--batch_files    数据预处理之后的批文件
  |--row_data       原始数据集（下载时什么样就什么样）
  |--saved_models   模型输出目录
  |--scripts        数据预处理脚本

## 文件类型说明
- *.fa：FASTA 文件（原始数据文件，即从官网下载的数据，不带标签）
- *.target：标签文件（one-hot 形式）
- *.tsv：FASTA 文件转的
- *.h5.batch1：批文件（按 batch size 压缩的），用于训练

# 数据预处理
## 数据来源

## 预处理过程
目标：下载的原始数据文件（*.fa） -> *.tsv -> HDF5 格式，数据 shape 为：(batch_size, 4, 1, len)

1. 运行 data_pre_process.py，将 *.fa 转 *.tsv

2. 运行 embedH5.py（命令行传参），将 *.tsv 转 *.h5.batch
```
python embedH5.py train.tsv train.target batch_file/train.h5
```

## 运行模型
