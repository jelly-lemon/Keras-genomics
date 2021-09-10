from models.BaseModel import BaseModel
from models.EasyModel import EasyModel
from models.Model1 import Model1


def get_model(model_name:str, save_dir:str, is_load_saved_model:bool=False) -> BaseModel:
    """
    获取模型

    :param model_name:
    :return:
    """
    if model_name == "EasyModel":
        return EasyModel(save_dir, is_load_saved_model)
    elif model_name == "Model1":
        return Model1(save_dir, is_load_saved_model)
    else:
        raise Exception("No model named: " + model_name)



