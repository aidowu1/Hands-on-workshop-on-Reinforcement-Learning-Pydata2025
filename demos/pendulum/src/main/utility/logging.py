import logging
import logging.handlers as handlers
import os

from src.main.configs import global_configs
from src.main.utility.utils import Helpers

class Logger(object):
    @staticmethod
    def __initialize(logger_root_path):
        """
        Initializes the logger
        """
        __logger = logging.getLogger(__name__)
        
        # Create the handlers
        c_handler = logging.StreamHandler()
        f_handler = handlers.RotatingFileHandler(logger_root_path, maxBytes=2000000, backupCount=20)
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)
        
        # Create formatters and add it to the handlers
        c_format = logging.Formatter(global_configs.LOGGING_FORMAT)
        f_format = logging.Formatter(global_configs.LOGGING_FORMAT)
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        if (__logger.hasHandlers()):
            __logger.handlers.clear()
        __logger.addHandler(c_handler)
        __logger.addHandler(f_handler)
        __logger.setLevel(logging.DEBUG)
        return __logger

    @staticmethod
    def getLogger(root_path="", logger_root_path=global_configs.LOG_PATH):
        """
        Gets the logger instance
        """
        Helpers.createDirectoryIfItDoesNotExist(global_configs.LOG_FOLDER)
        full_log_path = os.path.join(root_path, logger_root_path)
        logger = Logger.__initialize(full_log_path)
        return logger