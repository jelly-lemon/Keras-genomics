import os

data_dir = "/private/linmeng/ChIP-seq_690"

positive_total = 0

for dir_name in os.listdir(data_dir):
    sub_dir = data_dir + "/" + dir_name
    for file_name in os.listdir(sub_dir):
        file_path = sub_dir + "/" + file_name
        if file_path.find("train") == -1:
            continue
        # print(file_path)
        print(dir_name+"/"+file_name)

        negative_n = 0
        positive_n = 0
        with open(file_path) as file:
            line = file.readline()
            while line:
                line = line.replace("\n", "")
                line = line.split(" ")
                if line[2] == "0":
                    negative_n += 1
                elif line[2] == "1":
                    positive_n += 1
                line = file.readline()
        positive_total += positive_n
        print("%d\t%d\t%f" % (positive_n, negative_n, positive_n/negative_n))
print("positive total:", positive_total)