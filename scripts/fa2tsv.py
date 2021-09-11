def fa2tsv(fa_file_path, tsv_file_path):
    """
    .fa(FASTA) 文件转 .tsv 文件

    :param fa_file_path: .fa 文件输入路径
    :param tsv_file_path: .tsv 文件保存路径
    """
    with open(fa_file_path, mode="r") as fa_file, open(tsv_file_path, mode="w") as tsv_file:
        line1 = fa_file.readline()
        while line1 != "":
            line2 = fa_file.readline()
            if line2 != "":
                # 文本模式读取：\r\n 变成 \n
                # 去掉 line1 后面的换行符
                line1 = line1.split("\n")[0]
                tsv_file.write("%s\t%s" % (line1, line2))
            line1 = fa_file.readline()
        fa_file.close()
        tsv_file.close()




if __name__ == '__main__':
    fa2tsv("../row_data/test.fa", "../row_data/test.tsv")
