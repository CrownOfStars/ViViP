import torch
from datetime import datetime
import logging
import os
import os.path as osp

def get_datetime():
    cur_time = datetime.fromtimestamp(datetime.now().timestamp())
    str_date_time = cur_time.strftime("%Y-%m-%d,%H-%M-%S")
    return str_date_time

def init_logger(output_dir):
    
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",datefmt='%Y-%M-%D %X')

    logger = logging.getLogger('mylog')
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    osp.exists(output_dir),"{} not exist".format(output_dir)
    log_dir = osp.join(output_dir,get_datetime())
    os.mkdir(log_dir)
    file_handler=logging.FileHandler(filename=osp.join(log_dir,"log.txt"),mode='w',encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger,log_dir


if __name__ == "__main__":
    logger = init_logger('./output')
    logger.debug("hello world")
    logger.warning("warning")