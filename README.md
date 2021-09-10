基于 tensorflow/keras 对基因组数据（genomics data）进行深度学习训练、预测和超参数调整。

fork 自：https://github.com/gifford-lab/Keras-genomics

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

# 文件夹说明
--Keras-genomics
  |--batch_files    数据预处理之后的批文件
  |--row_data       原始数据集（下载时什么样就什么样）
  |--saved_models   模型输出目录
  |--scripts        数据预处理脚本

# 文件类型说明
- *.fa：FASTA 文件（原始数据文件，即从官网下载的数据，不带标签）
- *.target：标签文件（one-hot 形式）
- *.tsv：FASTA 文件转的
- *.h5.batch1：批文件（按 batch size 压缩的），用于训练

# 数据


# 运行过程
## 数据预处理
目标：下载的原始数据文件（*.fa） -> *.tsv -> HDF5 格式，数据 shape 为：(batch_size, 4, 1, len)

1. 运行 data_pre_process.py，将 *.fa 转 *.tsv

2. 运行 embedH5.py（命令行传参），将 *.tsv 转 *.h5.batch
```
python embedH5.py train.tsv train.target batch_file/train.h5
```

## 运行模型
