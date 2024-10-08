import logging
from logging.handlers import RotatingFileHandler

def create_rotating_logger(name: str, log_file: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5, level=logging.DEBUG) -> logging.Logger:
    """创建一个循环的 logger 实例。

    Args:
        name (str): logger 的名称。
        log_file (str): 日志文件的路径。
        max_bytes (int): 单个日志文件的最大字节数，默认为 10MB。
        backup_count (int): 保留的旧日志文件的数量，默认为 5。
        level (int): 日志级别，默认为 INFO

    Returns:
        logging.Logger: 配置好的循环 logger 实例。
    """
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
     # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 创建循环文件处理器
    if log_file:
        handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        handler.setLevel(level)
        handler.setFormatter(formatter)

    # 添加处理器到 logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger