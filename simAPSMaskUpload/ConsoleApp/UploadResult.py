from typing import TypeAlias

from simAPSMaskUpload.StorageReader.SeriesList import SeriesItem

__all__ = ["UploadResult", "UploadResults"]


class UploadResult:
    """单个序列的掩膜上传情况"""

    def __init__(self, series_info: SeriesItem, uploaded_mask_types: list[str]) -> None:
        """构造函数

        Args:
            series_info (_type_): _description_
            uploaded_mask_types (_type_): 成功上传的序列类型
        """
        self.series_info = series_info
        self.uploaded_mask_types = uploaded_mask_types


UploadResults: TypeAlias = list[UploadResult]
