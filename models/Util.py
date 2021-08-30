from BaseModel import BaseModel
from EasyModel import EasyModel


def get_model(model_name:str) -> BaseModel:
    """
    获取模型

    :param model_name:
    :return:
    """
    if model_name == "EasyModel":
        return EasyModel()
    else:
        raise Exception("No model named: " + model_name)



