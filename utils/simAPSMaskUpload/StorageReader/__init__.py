from os import path

# from simAPSMaskUpload.logger import logger
from .StorageReaderBase import StorageReaderBase
from .SqlcipherStorageReader import SqlcipherStorageReader

__all__ = [
    "check_storage_valid",
    "get_storage_obj",
]


def __check_connect_str_valid(connect_str: str) -> bool:
    """校验其他类型数据库连接字符串的有效性（暂时没做实现）

    Args:
        connect_str (str): 数据库连接字符串，不为空

    Raises:
        NotImplementedError: 未作其他数据库类型的实现

    Returns:
        bool: True 有效，可以正常连接且存在 series 表格；False 无效，不可以正常连接或不存在 series 表格
    """
    assert connect_str != None and connect_str.__len__() != 0
    # TODO: 暂时没做实现
    raise NotImplementedError(
        "暂时仅支持 sqlite 数据库，不支持连接其他类型数据库。连接字符串为 %s"
        % connect_str
    )


def check_storage_valid(storage_dir: str, connect_str: str | None) -> bool:
    """检查存储库参数有效性

    Args:
        storage_dir (str): 存储库目录路径
        connect_str (str): 数据库连接字符串，为空时默认使用 sqlite 连接

    Returns:
        bool: True 有效，False 无效
    """
    assert path.isdir(storage_dir)

    # 检查存储库路径有效性
    if not path.isdir(storage_dir + "/dcm"):
        logger.debug("存储库路径 %s 下不存在 dcm 目录" % storage_dir)
        return False

    # 检查数据库连接有效性
    if connect_str == None or connect_str.__len__() == 0:
        # 没有给连接字符串时，存储库目录下必须要有 sqlite3 文件夹，且存在 aview_data.db 文件
        sqlcipher_db_file = storage_dir + "/sqlite3/aview_data.db"
        if not path.exists(sqlcipher_db_file):
            logger.debug(
                "存储库路径 %s 下不存在 /sqlite3/aview_data.db 数据库文件" % storage_dir
            )
            return False
        if not SqlcipherStorageReader.check_valid(sqlcipher_db_file):
            return False
    else:
        # TODO: 进行其他类型数据库连接字符串的校验
        if not __check_connect_str_valid(connect_str):
            return False

    return True


def get_storage_obj(storage_dir: str, connect_str: str | None) -> StorageReaderBase:
    """返回存储库对象

    Args:
        storage_dir (str): 存储库目录路径
        connect_str (str): 数据库连接字符串，为空时默认使用 sqlite 连接

    Raises:
        NotImplementedError: 未作其他数据库类型的实现

    Returns:
        StorageReaderBase: 存储库对象
    """
    assert check_storage_valid(storage_dir, connect_str)

    # 若连接字符串不存在或为空，使用 sqlcipher 解析数据库
    if connect_str == None or connect_str.__len__() == 0:
        return SqlcipherStorageReader(storage_dir)
    else:
        # TODO：返回其他类型存储库对象，暂时未做实现
        raise NotImplementedError(
            "暂时仅支持 sqlite 数据库，不支持连接其他类型数据库。连接字符串为 %s"
            % connect_str
        )
