import logging
import os
from datetime import datetime

# 定义颜色（ANSI 转义码）
RESET = "\033[0m"
COLOR = {
    "DEBUG": "\033[37m",     # 灰色
    "INFO": "\033[36m",      # 青色
    "WARNING": "\033[33m",   # 黄色
    "ERROR": "\033[31m",     # 红色
    "CRITICAL": "\033[41m",  # 红底白字
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLOR.get(record.levelname, RESET)
        # 临时构造彩色的 levelname
        colored_levelname = f"{log_color}{record.levelname}{RESET}"
        original_levelname = record.levelname
        record.levelname = colored_levelname
        message = super().format(record)
        # 恢复原始值，避免污染 file_handler
        record.levelname = original_levelname
        return message

def get_logger(train_save):
    """

    Args:
        train_save (str): log文件要保存的目录

    Returns:
        (class): logger日志类方法
    """
    # 确保日志目录存在
    os.makedirs(train_save, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('Running_Logger')
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    
    now = datetime.now()
    file_handler = logging.FileHandler(
        fr'{train_save}/Running_{now.strftime("%Y_%m_%d_%H_%M")}.log', 
        encoding='utf-8'
        )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    #添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
