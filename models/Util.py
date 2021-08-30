from models.BaseModel import BaseModel
from models.EasyModel import EasyModel


def get_model(model_name:str, save_dir:str, save_tag:str) -> BaseModel:
    """
    获取模型

    :param model_name:
    :return:
    """
    if model_name == "EasyModel":
        return EasyModel(save_dir, save_tag)
    else:
        raise Exception("No model named: " + model_name)



