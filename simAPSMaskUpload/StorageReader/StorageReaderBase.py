from abc import abstractmethod
from os import path

from simAPSMaskUpload.StorageReader.SeriesList import SeriesList

__all__ = ["StorageReaderBase"]


class StorageReaderBase:
    """AVIEW 存储库读取基类"""

    def __init__(self, storage_dir: str) -> None:
        """AVIEW 存储库读取基类构造

        Args:
            storage_dir (str): 存储库路径
        """
        assert path.isdir(storage_dir)
        self._storage_dir = storage_dir

    @abstractmethod
    def get_all_series_list(self) -> SeriesList:
        """获取存储库所有的序列信息

        Returns:
            SeriesList: 序列信息列表对象
        """
        pass

    @abstractmethod
    def get_series_list_by_patient_ids(self, patient_ids: list[str]) -> SeriesList:
        """通过 patient id 列表获取序列信息

        Args:
            patient_ids (str): patient id 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        # assert patient_ids.__len__() != 0
        pass

    @abstractmethod
    def get_series_list_by_study_ids(self, study_ids: list[str]) -> SeriesList:
        """通过 study id 列表获取序列信息

        Args:
            study_ids (list[str]): study id 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        # assert study_ids.__len__() != 0
        pass

    @abstractmethod
    def get_series_list_by_access_nums(self, access_nums: list[str]) -> SeriesList:
        """通过 access num 列表获取序列信息

        Args:
            access_nums (list): access num 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        # assert access_nums.__len__() != 0
        pass

    @abstractmethod
    def get_series_list_by_study_uids(self, study_uids: list[str]) -> SeriesList:
        """通过 study uid 列表获取序列信息

        Args:
            study_uids (list[str]): study instance uid 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        # assert study_uids.__len__() != 0
        pass

    @abstractmethod
    def get_series_list_by_series_uids(self, series_uids: list[str]) -> SeriesList:
        """通过 series uid 列表获取序列信息

        Args:
            series_uids (list[str]): series instance uid 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        # assert series_uids.__len__() != 0
        pass

    @abstractmethod
    def get_series_list_by_aps_tags(self, aps_tags: list[str]) -> SeriesList:
        """通过 aps 标签获取序列信息（这些序列都成功运行指定的 aps 并生成相应掩膜结果）

        Args:
            aps_tags (list[str]): aps 标签列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        # assert aps_tags.__len__() != 0
        pass

    @abstractmethod
    def get_db_id_by_series_uid(self, series_uid: str) -> tuple[int, int] | None:
        """通过序列uid获取相应的检查数据库id和序列数据库id

        Args:
            series_uid (str): 序列uid

        Returns:
            tuple[int, int]: 检查数据库id，序列数据库id
        """
        # assert series_uid.__len__()!=0
        pass
