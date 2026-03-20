from os import path, mkdir
from logging import getLogger
from logging.config import fileConfig


from simAPSMaskUpload.settings import PROJECT_NAME, DEFAULT_LOGGER_CONFIG_FILE

__all__ = ["logger"]


if not path.isdir("logs"):
    mkdir("logs")
# fileConfig(DEFAULT_LOGGER_CONFIG_FILE, encoding="utf-8")
logger = getLogger(PROJECT_NAME)