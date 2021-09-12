import sys


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message:str):
        # 训练模型输出时会有很多 \b 输出到文件，这里给替换掉
        message = message.replace("\b", "")
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass