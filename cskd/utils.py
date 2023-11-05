import os
import io
import torch
import logging
from time import time
from functools import lru_cache
from balls.imgproc import imdecode
from importlib.util import spec_from_file_location, module_from_spec

__all__ = [
    "load_config_from_file",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_ema",
    "get_logger",
    "Clock",
]


def load_config_from_file(config_file, class_name="Config"):
    spec = spec_from_file_location("config", config_file)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    config_class = getattr(m, class_name)
    return config_class

def save_checkpoint(checkpoint_dir, checkpoint_name, checkpoint):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    try:
        torch.save(checkpoint, checkpoint_path)
        return True
    except:
        return False

def load_checkpoint(checkpoint_dir, checkpoint_name):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint
    except:
        return None

def load_model_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

@lru_cache(maxsize=1)
def get_logger(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=config.log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


class Clock:
    def __init__(self, ndigits=3):
        self.start_time = 0
        self.stop_time = 0
        self.checkpoint_time = 0
        self.total_time = 0
        self.ndigits = ndigits

    def start(self):
        self.start_time = time()
        self.checkpoint_time = self.start_time

    def lap(self):
        lap_time = round(time() - self.checkpoint_time, self.ndigits)
        self.checkpoint_time = time()
        return lap_time

    def stop(self):
        self.stop_time = time()
        self.total_time = round(self.stop_time - self.start_time, self.ndigits)

    def get(self):
        return self.total_time

    def reset(self):
        self.__init__(ndigits=self.ndigits)
