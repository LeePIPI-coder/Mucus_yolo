from typing import TypeAlias
from os import path

__all__ = [
    "Position",
    "SeriesItem",
    "SeriesList",
]


class Position:
    """患者坐标信息"""

    def __init__(self, X: float, Y: float, Z: float) -> None:
        """构造函数

        Args:
            X (float): IN 患者坐标系 X
            Y (float): IN 患者坐标系 Y
            Z (float): IN 患者坐标系 Z
        """
        self.X = X
        self.Y = Y
        self.Z = Z

    def __str__(self) -> str:
        return "(%f, %f, %f)" % (self.X, self.Y, self.Z)


class SeriesItem:
    """序列信息"""

    def __init__(
        self,
        patient_id: str,
        patient_name: str,
        study_id: str,
        access_number: str,
        study_instance_uid: str,
        series_number: int,
        series_instance_uid: str,
        abs_path: str,
        patient_pos_max: Position,
        patient_pos_min: Position,
        rows: int,
        cols: int,
        sop_instance_count: int,
        updated_at: str,
        study_date_time: str,
        sopInstanceUid:str=None,
        top_left:tuple=None,
        bottom_right:tuple=None,
        # patient_vol: float=None
    ) -> None:
        """构造函数

        Args:
            patient_id (str): IN 患者ID
            patient_name (str): IN 患者姓名
            study_id (str): IN 检查ID
            access_number (str): IN 注册号
            study_instance_uid (str): IN 检查实例UID
            series_number (int): IN 序列编号
            series_instance_uid (str): IN 序列实例UID
            abs_path (str): IN 序列存放的绝对路径
        """
        assert study_instance_uid.__len__() != 0
        assert series_instance_uid.__len__() != 0
        assert path.isdir(abs_path)
        assert rows > 0 and cols > 0 and sop_instance_count > 0
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.study_id = study_id
        self.access_number = access_number
        self.study_instance_uid = study_instance_uid
        self.series_number = series_number
        self.series_instance_uid = series_instance_uid
        self.abs_path = abs_path
        self.patient_pos_max = patient_pos_max
        self.patient_pos_min = patient_pos_min
        self.rows = rows
        self.cols = cols
        self.sop_instance_count = sop_instance_count
        self.updated_at = updated_at
        self.study_date_time = study_date_time
        self.sopInstanceUid = sopInstanceUid
        self.top_left= top_left
        self.bottom_right = bottom_right
        # self.patient_vol = patient_vol


SeriesList: TypeAlias = list[SeriesItem]
