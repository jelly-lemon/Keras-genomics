import json

from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.saving.model_config import model_from_json

from models.BaseModel import BaseModel
from models.EasyModel import EasyModel
from models.Model1 import Model1


def load_saved_model(architecture_path:str, weight_path:str, params_path:str) -> Model:
    """
    加载保存的模型

    :param architecture_path:
    :param weight_path:
    :param params_path:
    :return:
    """
    # 加载超参数
    params = json.load(open(params_path))

    # 加载模型结构
    model = model_from_json(open(architecture_path).read())

    # 加载保存的权重
    model.load_weights(weight_path)

    return model

def get_base_model(model_name:str, save_dir:str) -> BaseModel:
    """
    获取模型

    :param model_name:
    :return:
    """
    if model_name == "EasyModel":
        return EasyModel(save_dir)
    elif model_name == "Model1":
        return Model1(save_dir)
    else:
        raise Exception("No model named: " + model_name)



