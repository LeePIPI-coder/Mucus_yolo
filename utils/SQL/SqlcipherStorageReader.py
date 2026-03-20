from os import path
from sqlcipher3 import connect, Connection
from re import match

from ..simAPSMaskUpload.settings import (
    AVIEW_SQLITE_KEY,
    AVIEW_SQLITE_CIPHER,
    AVIEW_SQLITE_KDF_ITER,
    AVIEW_SQLITE_CIPHER_PAGE_SIZE,
)
from ..simAPSMaskUpload.StorageReader.StorageReaderBase import StorageReaderBase
from ..simAPSMaskUpload.StorageReader.SeriesList import SeriesItem, SeriesList, Position


__all__ = ["SqlcipherStorageReader"]

image_position_patient_pattern = r"^([eE0-9.+-]*)\\([eE0-9.+-]*)\\([eE0-9.+-]*)$"
pixel_spacing_pattern = r"^([eE0-9.+-]*)\\([eE0-9.+-]*)$"


class SqlcipherStorageReader(StorageReaderBase):
    """AVIEW 的 sqlite 类型数据存储库读取

    Args:
        StorageReaderBase (_type_): AVIEW 数据库读取接口
    """

    def __init__(self, storage_dir: str) -> None:
        """构造函数

        Args:
            storage_dir (str): AVIEW 的存储库路径
        """
        super().__init__(storage_dir)

        storage_db_file = storage_dir + "/sqlite3/aview_data.db"
        assert path.isfile(storage_db_file)
        assert SqlcipherStorageReader.check_valid(storage_db_file)

        # 连接数据库
        self.__conn = SqlcipherStorageReader.__connect_sqlcipher(storage_db_file)

    def __del__(self):
        """析构函数"""
        self.__conn.close()

    def __get_all_patient(self) -> list[tuple[int, str, str]]:
        """获取所有患者信息

        Returns:
            list[tuple[int, str, str]]: 患者信息列表 (patient_db_id, patient_id, patient_name)
        """
        cur = self.__conn.cursor()

        patient_list: list[tuple[int, str, str]] = []
        cur.execute("SELECT id, patient_id, patient_name, updated_at FROM patient;")
        for patient_db_id, patient_id, patient_name, updated_at in cur.fetchall():
            patient_list.append((patient_db_id, patient_id, patient_name, updated_at))
        return patient_list

    def __get_studies_by_patient_db_id(
        self, patient_db_id: int
    ) -> list[tuple[int, str, str, str]]:
        """获取患者相关检查信息

        Args:
            patient_db_id (int): 患者数据库ID

        Returns:
            list[tuple[int, str, str]]: 检查信息列表 (study_db_id, study_id, accession_number, study_instance_uid)
        """
        cur = self.__conn.cursor()
        study_list: list[tuple[int, str, str, str]] = []
        cur.execute(
            "SELECT id, study_id, accession_number, study_instance_uid FROM study "
            "WHERE fk_patient_id = ?;",
            (patient_db_id,),
        )
        for (
            study_db_id,
            study_id,
            accession_number,
            study_instance_uid,
        ) in cur.fetchall():
            study_list.append(
                (study_db_id, study_id, accession_number, study_instance_uid)
            )
        return study_list

    def __get_series_by_study_db_id(
        self, study_db_id: int
    ) -> list[tuple[int, int, str, str, int]]:
        """获取检查相关序列信息

        Args:
            study_db_id (int): 检查数据库ID

        Returns:
            list[tuple[int, int, str, str, int]]: 序列信息列表 (series_db_id, series_number, series_instance_uid, workspace_path, sop_instance_count)
        """
        cur = self.__conn.cursor()
        series_list: list[tuple[int, int, str, str, int]] = []
        cur.execute(
            "SELECT id, series_number, series_instance_uid, workspace_path, sop_instance_count FROM series "
            "WHERE fk_study_id = ? "
            "AND modality = 'CT';",
            (study_db_id,),
        )
        for (
            series_db_id,
            series_number,
            series_instance_uid,
            workspace_path,
            sop_instance_count,
        ) in cur.fetchall():
            series_list.append(
                (
                    series_db_id,
                    series_number,
                    series_instance_uid,
                    workspace_path,
                    sop_instance_count,
                )
            )
        return series_list

    def __get_position_from_DB_str(self, position_str: str) -> Position | None:
        """从数据字段 image_position_patient 中解析切片所在的患者坐标

        Args:
            first_position (str): 类似于 X\\Y\\Z 格式的数据字段 (-181.700\\-200.500\\-4.750)

        Returns:
            Position | None: 患者坐标对象 | 坐标字符串无效
        """
        if position_str.__len__() == 0:
            return None

        m = match(image_position_patient_pattern, position_str)
        if m == None:
            return None
        pos_tuple = m.groups()
        if pos_tuple.__len__() != 3:
            return None
        return Position(float(pos_tuple[0]), float(pos_tuple[1]), float(pos_tuple[2]))

    def __get_pixel_spacing(self, pixel_spacing_str: str) -> tuple[float, float] | None:
        """从数据字段 pixel_spacing 中解析切片的像素间距

        Args:
            pixel_spacing_str (str): 类似于 X\\Y 格式的数据字段

        Returns:
            tuple[float, float] | None: 像素间距 | 像素间距字符串无效
        """
        if pixel_spacing_str.__len__() == 0:
            return None

        m = match(pixel_spacing_pattern, pixel_spacing_str)
        if m == None:
            return None
        spacing_tuple = m.groups()
        if spacing_tuple.__len__() != 2:
            return None
        return float(spacing_tuple[0]), float(spacing_tuple[1])

    def __get_instance_by_series_db_id(
        self, series_db_id: int
    ) -> tuple[Position, Position, int, int] | None:
        """通过序列数据库ID获取切片信息

        Args:
            series_db_id (int): IN 序列数据库ID

        Returns:
            tuple[Position, Position, int, int] | None: 切片信息 (patient_pos_max, patient_pos_min, rows, cols) | 序列数据库ID 不存在，无法获取切片信息
        """
        # 读取每一张切片的 image_position_patient, rows, columns, pixel_spacing 信息
        cur = self.__conn.cursor()
        cur.execute(
            "SELECT id, image_position_patient, rows, columns, pixel_spacing, slice_thickness FROM sop_instance "
            "WHERE fk_series_id = ?;",
            (series_db_id,),
        )

        first_result = cur.fetchone()
        if first_result == None:
            logger.warning("通过序列数据库ID %d 没有获取到切片信息" % (series_db_id))
            return None

        # 比较计算获得 patient_pos_max 和 patient_pos_min 信息
        (
            first_instance_db_id,
            first_position_str,
            first_rows,
            first_cols,
            first_pixel_spacing_str,
            first_slice_thickness,
        ) = first_result
        position_init = self.__get_position_from_DB_str(first_position_str)
        if position_init == None:
            # logger.error(
            #     "通过序列数据库ID %d 获取到的切片数据库ID %d 的患者坐标信息 %s 无效"
            #     % (series_db_id, first_instance_db_id, first_position_str)
            # )
            return None
        position_min = Position(position_init.X, position_init.Y, position_init.Z)
        pixel_spacing_tuple = self.__get_pixel_spacing(first_pixel_spacing_str)
        if pixel_spacing_tuple == None:
            logger.error(
                "通过序列数据库ID %d 获取到的切片数据库ID %d 的像素间距信息 %s 无效"
                % (series_db_id, first_instance_db_id, first_pixel_spacing_str)
            )
            return None
        pixel_spacing_X, pixel_spacing_Y = pixel_spacing_tuple

        position_max = Position(
            position_init.X + first_rows * pixel_spacing_X,
            position_init.Y + first_cols * pixel_spacing_Y,
            position_init.Z,
        )

        for (
            instance_db_id,
            position_str,
            rows,
            cols,
            pixel_spacing_str,
            slice_thickness,
        ) in cur.fetchall():
            # 确保每一张切片的 rows，cols 保持一致
            # fmt: off
            if rows != first_rows or cols != first_cols:
                logger.error(
                    "通过序列数据库ID %d 获取到的切片数据库ID %d 的行数 %d ,列数 %d与初始切片数据库ID %d 行数 %d, 列数 %d 存在不同"
                    % (
                        series_db_id,
                        instance_db_id, rows, cols,
                        first_instance_db_id, first_rows, first_cols,
                    )
                )
                return None
            if pixel_spacing_str != first_pixel_spacing_str:
                logger.error(
                    "通过序列数据库ID %d 获取到的切片数据库ID %d 的像素间距 %s 与初始切片数据库ID %d 的像素间距 %s 存在不同"
                    % (
                        series_db_id,
                        instance_db_id, pixel_spacing_str,
                        first_instance_db_id, first_pixel_spacing_str,
                    )
                )
                return None
            if slice_thickness != first_slice_thickness:
                logger.error(
                    "通过序列数据库ID %d 获取到的切片数据库ID %d 的切片厚度 %f 与初始切片数据库ID %d 的像素间距 %f 存在不同"
                    % (
                        series_db_id,
                        instance_db_id, slice_thickness,
                        first_instance_db_id, first_slice_thickness,
                    )
                )
                return None
            # fmt: on
            position = self.__get_position_from_DB_str(position_str)
            if position == None:
                logger.error(
                    "通过序列数据库ID %d 获取到的切片数据库ID %d 的患者坐标信息 %s 无效"
                    % (series_db_id, instance_db_id, position_str)
                )
                return None
            if position.Z < position_min.Z:
                position_min.Z = position.Z
            if position.Z > position_max.Z:
                position_max.Z = position.Z

        # 追加层厚的高度
        position_max.Z += first_slice_thickness
        return position_max, position_min, first_rows, first_cols

    def __append_series_by_study_db_id(
        self,
        series_info_list: SeriesList,
        study_db_id: int,
        patient_id: str,
        patient_name: str,
        study_id: str,
        accession_number: str,
        study_instance_uid: str,
        updated_at,
    ) -> None:
        """通过检查数据库ID添加序列信息

        Args:
            series_info_list (SeriesList): OUT 序列信息列表
            study_db_id (int): IN 检查数据库ID
            patient_id (str): IN 患者ID
            patient_name (str): IN 患者姓名
            study_id (str): IN 检查ID
            accession_number (str): IN 注册号
            study_instance_uid (str): IN 检查UID
        """
        # 获取检查对应的序列信息
        for (series_db_id,series_number,series_instance_uid,workspace_path,sop_instance_count,) in self.__get_series_by_study_db_id(study_db_id):
            instance_info = self.__get_instance_by_series_db_id(series_db_id)
            if instance_info == None:
                continue
            patient_pos_max, patient_pos_min, rows, cols = instance_info

            series_info_list.append(
                SeriesItem(
                    patient_id,
                    patient_name,
                    study_id,
                    accession_number,
                    study_instance_uid,
                    series_number,
                    series_instance_uid,
                    self._storage_dir + "/" + workspace_path,
                    # patient_pos_max,
                    # patient_pos_min,
                    rows,
                    cols,
                    sop_instance_count,
                    updated_at
                )
            )

    def get_all_series_list(self) -> SeriesList:
        """获取存储库所有的序列信息

        Returns:
            SeriesList: IN 序列信息列表对象
        """
        series_info_list: SeriesList = []

        # 获取所有患者信息
        for (patient_db_id, patient_id, patient_name, updated_at) in self.__get_all_patient():
            # 获取患者对应的检查信息
            for (study_db_id, study_id, accession_number, study_instance_uid,) in self.__get_studies_by_patient_db_id(patient_db_id):
                # 获取检查对应的序列信息
                self.__append_series_by_study_db_id(
                    series_info_list,
                    study_db_id,
                    patient_id,
                    patient_name,
                    study_id,
                    accession_number,
                    study_instance_uid,
                    updated_at
                )
        return series_info_list

    def __get_patients_by_patient_ids(
        self, patient_ids: list[str]
    ) -> list[tuple[int, str, str]]:
        """根据患者ID列表获取相应患者信息

        Args:
            patient_ids (list[str]): 患者ID列表

        Returns:
            list[tuple[int, str, str]]: 患者信息 (patient_db_id, patient_id, patient_name)
        """
        cur = self.__conn.cursor()
        patient_list: list[tuple[int, str, str]] = []
        for patient_id in patient_ids:
            # fmt: off
            cur.execute(
                "SELECT id, patient_name FROM patient "
                "WHERE patient_id = ?;",
                (patient_id,),
            )
            # fmt on
            for patient_db_id, patient_name in cur.fetchall():
                patient_list.append((patient_db_id, patient_id, patient_name))
        return patient_list

    def get_series_list_by_patient_ids(self, patient_ids: list[str]) -> SeriesList:
        """通过 patient id 列表获取序列信息

        Args:
            patient_ids (str): patient id 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        assert patient_ids.__len__() != 0
        series_info_list: SeriesList = []

        # 获取相应的患者信息
        for (
            patient_db_id,
            patient_id,
            patient_name,
        ) in self.__get_patients_by_patient_ids(patient_ids):
            # 获取患者对应的检查信息
            for (
                study_db_id,
                study_id,
                accession_number,
                study_instance_uid,
            ) in self.__get_studies_by_patient_db_id(patient_db_id):
                # 获取检查对应的序列信息
                self.__append_series_by_study_db_id(
                    series_info_list,
                    study_db_id,
                    patient_id,
                    patient_name,
                    study_id,
                    accession_number,
                    study_instance_uid,
                    updated_at
                )
        return series_info_list

    def __get_studies_by_study_ids(
        self, study_ids: list[str]
    ) -> list[tuple[int, str, str, str, int]]:
        """根据检查ID列表获取相应检查信息

        Args:
            study_ids (list[str]): 检查ID列表

        Returns:
            list[tuple[int, str, str, int]]: 检查信息 (study_db_id, study_id, accession_number, study_instance_uid, patient_db_id)
        """
        cur = self.__conn.cursor()
        study_list: list[tuple[int, str, str, str, int]] = []
        for study_id in study_ids:
            cur.execute(
                "SELECT id, accession_number, study_instance_uid, fk_patient_id FROM study "
                "WHERE study_id = ?;",
                (study_id,),
            )
            for (
                study_db_id,
                accession_number,
                study_instance_uid,
                patient_db_id,
            ) in cur.fetchall():
                study_list.append(
                    (
                        study_db_id,
                        study_id,
                        accession_number,
                        study_instance_uid,
                        patient_db_id,
                    )
                )
        return study_list

    def __get_patient_by_patient_db_id(
        self, patient_db_id: int
    ) -> tuple[str, str] | None:
        """根据患者数据库ID获取患者信息

        Args:
            patient_db_id (int): 患者数据库ID

        Returns:
            tuple[str, str]: 患者信息 (patient_id, patient_name) | 患者数据库ID不存在，无法获取患者信息
        """
        cur = self.__conn.cursor()
        # fmt: off
        cur.execute(
            "SELECT patient_id, patient_name FROM patient "
            "WHERE id = ?;",
            (patient_db_id,),
        )
        # fmt: on
        patient_info = cur.fetchone()
        if patient_info == None:
            logger.warning("通过 患者数据库ID %d 没有获取到患者信息" % patient_db_id)
            return None
        return patient_info

    def get_series_list_by_study_ids(self, study_ids: list[str]) -> SeriesList:
        """通过 study id 列表获取序列信息

        Args:
            study_ids (list[str]): study id 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        assert study_ids.__len__() != 0
        series_info_list: SeriesList = []

        # 获取相应的检查信息
        for (
            study_db_id,
            study_id,
            accession_number,
            study_instance_uid,
            patient_db_id,
        ) in self.__get_studies_by_study_ids(study_ids):
            # 获取检查对应的患者信息
            patient_info = self.__get_patient_by_patient_db_id(patient_db_id)
            if patient_info == None:
                continue
            patient_id, patient_name = patient_info
            # 获取检查对应的序列信息
            self.__append_series_by_study_db_id(
                series_info_list,
                study_db_id,
                patient_id,
                patient_name,
                study_id,
                accession_number,
                study_instance_uid,
            )
        return series_info_list

    def __get_studies_by_access_nums(
        self, access_nums: list[str]
    ) -> list[tuple[int, str, str, str, int]]:
        """通过 access_num 获取检查信息

        Args:
            access_nums (list[str]): access num 列表

        Returns:
            List[tuple[int, str, str, str, int]]: 检查信息 (study_db_id, study_id, accession_number, study_instance_uid, patient_db_id)
        """
        cur = self.__conn.cursor()
        study_list: list[tuple[int, str, str, str, int]] = []
        for access_num in access_nums:
            cur.execute(
                "SELECT id, study_id, study_instance_uid, fk_patient_id FROM study "
                "WHERE accession_number = ?;",
                (access_num,),
            )
            for (
                study_db_id,
                study_id,
                study_instance_uid,
                patient_db_id,
            ) in cur.fetchall():
                study_list.append(
                    (
                        study_db_id,
                        study_id,
                        access_num,
                        study_instance_uid,
                        patient_db_id,
                    )
                )
        return study_list

    def get_series_list_by_access_nums(self, access_nums: list[str]) -> SeriesList:
        """通过 access num 列表获取序列信息

        Args:
            access_nums (list): access num 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        assert access_nums.__len__() != 0
        series_info_list: SeriesList = []

        # 获取相应的检查信息
        for (
            study_db_id,
            study_id,
            accession_number,
            study_instance_uid,
            patient_db_id,
        ) in self.__get_studies_by_access_nums(access_nums):
            # 获取检查对应的患者信息
            patient_info = self.__get_patient_by_patient_db_id(patient_db_id)
            if patient_info == None:
                continue
            patient_id, patient_name = patient_info
            # 获取检查对应的序列信息
            self.__append_series_by_study_db_id(
                series_info_list,
                study_db_id,
                patient_id,
                patient_name,
                study_id,
                accession_number,
                study_instance_uid,
            )
        return series_info_list

    def __get_studies_by_study_uids(
        self, study_uids: list[str]
    ) -> list[tuple[int, str, str, str, int]]:
        """通过 检查uid 获取相应检查信息

        Args:
            study_uids (list[str]): 检查uid列表

        Returns:
            list[tuple[int, str, str, str, int]]: 检查信息 (study_db_id, study_id, accession_number, study_instance_uid, patient_db_id)
        """
        cur = self.__conn.cursor()
        study_list: list[tuple[int, str, str, str, int]] = []
        for study_uid in study_uids:
            cur.execute(
                "SELECT id, study_id, accession_number, fk_patient_id FROM study "
                "WHERE study_instance_uid = ?;",
                (study_uid,),
            )
            for (
                study_db_id,
                study_id,
                access_num,
                patient_db_id,
            ) in cur.fetchall():
                study_list.append(
                    (
                        study_db_id,
                        study_id,
                        access_num,
                        study_uid,
                        patient_db_id,
                    )
                )
        return study_list

    def get_series_list_by_study_uids(self, study_uids: list[str]) -> SeriesList:
        """通过 study uid 列表获取序列信息

        Args:
            study_uids (list[str]): study instance uid 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        # assert study_uids.__len__() != 0
        series_info_list: SeriesList = []

        # 获取相应的检查信息
        for (
            study_db_id,
            study_id,
            accession_number,
            study_instance_uid,
            patient_db_id,
        ) in self.__get_studies_by_study_uids(study_uids):
            # 获取检查对应的患者信息
            patient_info = self.__get_patient_by_patient_db_id(patient_db_id)
            if patient_info == None:
                continue
            patient_id, patient_name = patient_info
            # 获取检查对应的序列信息
            self.__append_series_by_study_db_id(
                series_info_list,
                study_db_id,
                patient_id,
                patient_name,
                study_id,
                accession_number,
                study_instance_uid,
            )
        return series_info_list

    def __get_series_by_series_uids(
        self,
        series_uids: list[str],
    ) -> list[tuple[int, int, str, str, int, int, int]]:
        """通过 序列uid 获取相应的序列信息

        Args:
            series_uids (list[str]): 序列uid列表。正常情况下

        Returns:
            list[tuple[int, int, str, str, int, int, int]]: 序列信息 (series_db_id, series_number, series_instance_uid, workspace_path, sop_instance_count, patient_db_id, study_db_id)
        """
        cur = self.__conn.cursor()
        series_list: list[tuple[int, int, str, str, int, int, int]] = []

        for series_uid in series_uids:
            cur.execute(
                "SELECT id, series_number, workspace_path, sop_instance_count, fk_patient_id, fk_study_id FROM series "
                "WHERE series_instance_uid = ? "
                "AND modality = 'CT';",
                (series_uid,),
            )

            for (
                series_db_id,
                series_number,
                workspace_path,
                sop_instance_count,
                patient_db_id,
                study_db_id,
            ) in cur.fetchall():
                series_list.append(
                    (
                        series_db_id,
                        series_number,
                        series_uid,
                        workspace_path,
                        sop_instance_count,
                        patient_db_id,
                        study_db_id,
                    )
                )
        return series_list

    def __get_study_by_study_db_id(
        self, study_db_id: int
    ) -> tuple[str, str, str] | None:
        """通过 检查数据库ID 获取检查信息

        Args:
            study_db_id (int): 检查数据库ID

        Returns:
            tuple[str, str, str] | None: 检查信息 (study_id, accession_number, study_instance_uid) | 检查数据库ID不存在，无法获取检查信息
        """
        cur = self.__conn.cursor()
        cur.execute(
            "SELECT study_id, accession_number, study_instance_uid FROM study "
            "WHERE id = ?;",
            (study_db_id,),
        )
        study_info = cur.fetchone()
        if study_info == None:
            logger.warning("通过 检查数据库ID %d 没有获取到检查信息" % study_db_id)
            return None
        return study_info

    def __append_series_info(
        self,
        patient_db_id: int,
        study_db_id: int,
        series_db_id: int,
        series_number: int,
        series_instance_uid: str,
        workspace_path: str,
        sop_instance_count: int,
        series_info_list: SeriesList,
    ) -> None:
        """添加序列信息到列表中

        Args:
            patient_db_id (int): IN 患者数据库ID
            study_db_id (int): IN 检查数据库ID
            series_db_id (int): IN 序列数据库ID
            series_number (int): IN 序列编号
            series_instance_uid (str): IN 序列UID
            workspace_path (str): IN 序列工作路径
            sop_instance_count (int): IN 序列实例数
            series_info_list (SeriesList): IN OUT 序列信息列表
        """
        # 获取序列对应的患者信息
        patient_info = self.__get_patient_by_patient_db_id(patient_db_id)
        if patient_info == None:
            return
        patient_id, patient_name = patient_info
        # 获取序列对应的检查信息
        study_info = self.__get_study_by_study_db_id(study_db_id)
        if study_info == None:
            return
        study_id, accession_number, study_instance_uid = study_info
        # 获取序列对应的切片信息
        instance_info = self.__get_instance_by_series_db_id(series_db_id)
        if instance_info == None:
            return
        patient_pos_max, patient_pos_min, rows, cols = instance_info
        series_info_list.append(
            SeriesItem(
                patient_id,
                patient_name,
                study_id,
                accession_number,
                study_instance_uid,
                series_number,
                series_instance_uid,
                self._storage_dir + "/" + workspace_path,
                patient_pos_max,
                patient_pos_min,
                rows,
                cols,
                sop_instance_count,
            )
        )
        return

    def get_series_list_by_series_uids(self, series_uids: list[str]) -> SeriesList:
        """通过 series uid 列表获取序列信息

        Args:
            series_uids (list[str]): series instance uid 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        assert series_uids.__len__() != 0
        series_info_list: SeriesList = []

        # 获取相应的序列信息
        for (
            series_db_id,
            series_number,
            series_instance_uid,
            workspace_path,
            patient_db_id,
            study_db_id,
            sop_instance_count,
        ) in self.__get_series_by_series_uids(series_uids):
            self.__append_series_info(
                patient_db_id,
                study_db_id,
                series_db_id,
                series_number,
                series_instance_uid,
                workspace_path,
                sop_instance_count,
                series_info_list,
            )
        return series_info_list

    @staticmethod
    def __connect_sqlcipher(sqlcipher_db_file: str) -> Connection:
        """连接 sqlcipher 数据库，仅类内调用

        Args:
            sqlcipher_db_file (str): 数据库文件路径

        Returns:
            Connection: 连接对象
        """
        conn = connect(sqlcipher_db_file)
        cur = conn.cursor()
        cur.execute("PRAGMA key = '" + AVIEW_SQLITE_KEY + "';")
        # 这一句必不可少，否则无法访问；在 PRAGMA 第一行和最后一行都不行
        cur.execute("PRAGMA cipher_compatibility = 3;")
        cur.execute("PRAGMA cipher = '" + AVIEW_SQLITE_CIPHER + "';")
        cur.execute("PRAGMA kdf_iter = " + str(AVIEW_SQLITE_KDF_ITER) + ";")
        cur.execute(
            "PRAGMA cipher_page_size = " + str(AVIEW_SQLITE_CIPHER_PAGE_SIZE) + ";"
        )

        # 以 wal 模式访问 sqlite 数据库
        # AVIEW 默认以 wal 模式访问
        mode = cur.execute("PRAGMA journal_mode;").fetchone()[0]
        if mode != "wal":
            cur.execute("PRAGMA journal_mode=WAL;")
        return conn

    @staticmethod
    def __check_table_and_columns_existed(
        sqlcipher_db_file: str,
        conn: Connection,
        table_name: str,
        col_type_dict: dict[str, str],
    ) -> bool:
        """检查表是否存在且包含指定字段

        Args:
            sqlcipher_db_file (str): 数据库文件，日志打印用
            conn (Connection): 数据库连接
            table_name (str): 表名称
            col_type_dict (dict[str, str]): 字段名称和类型

        Returns:
            bool: True 存在，False 不存在
        """
        cur = conn.cursor()
        # fmt: off
        cur.execute(
            "SELECT * FROM sqlite_master "
            "WHERE type = 'table' "
            "AND name = ?;",
            (table_name,),
        )
        # fmt: on
        if not cur.fetchone():
            logger.error(
                "数据库文件 %s 中不存在数据表 %s" % (sqlcipher_db_file, table_name)
            )
            return False

        for col, col_type in col_type_dict.items():
            # fmt: off
            cur.execute(
                "SELECT * FROM pragma_table_info(?) "
                "WHERE name = ? "
                "AND type = ?;",
                (table_name, col, col_type),
            )
            # fmt: on
            if not cur.fetchone():
                logger.error(
                    "数据库文件 %s 中的数据表 %s 中不存在字段 %s 或字段类型不是 %s"
                    % (sqlcipher_db_file, table_name, col, col_type)
                )
                return False

        return True

    @staticmethod
    def check_valid(sqlcipher_db_file: str) -> bool:
        """判断 sqlcipher 数据库文件有效性

        Args:
            sqlcipher_db_file (str): sqlcipher 数据库文件路径

        Returns:
            bool: True 有效，存在相应结构表格；False 无效，无法读取或表结构无效
        """
        assert path.isfile(sqlcipher_db_file)

        # 判断是否能够读取db文件
        try:
            conn = SqlcipherStorageReader.__connect_sqlcipher(sqlcipher_db_file)
        except Exception as e:
            logger.error(
                "无法连接数据库文件 %s，错误原因为 %s" % (sqlcipher_db_file, e)
            )
            return False

        valid_status = True
        SQLITE_TEXT = "TEXT"
        SQLITE_INTEGER = "INTEGER"
        # 判断 patient 表是否存在，且表结构有效
        if not SqlcipherStorageReader.__check_table_and_columns_existed(
            sqlcipher_db_file,
            conn,
            "patient",
            {
                "id": SQLITE_INTEGER,
                "patient_id": SQLITE_TEXT,
                "patient_name": SQLITE_TEXT,
            },
        ):
            valid_status = False

        # 判断 study 表是否存在，且表结构有效
        if not SqlcipherStorageReader.__check_table_and_columns_existed(
            sqlcipher_db_file,
            conn,
            "study",
            {
                "id": SQLITE_INTEGER,
                "study_id": SQLITE_TEXT,
                "study_instance_uid": SQLITE_TEXT,
                "fk_patient_id": SQLITE_INTEGER,
            },
        ):
            valid_status = False

        # 判断 series 表是否存在，且表结构有效
        if not SqlcipherStorageReader.__check_table_and_columns_existed(
            sqlcipher_db_file,
            conn,
            "series",
            {
                "series_number": SQLITE_INTEGER,
                "series_instance_uid": SQLITE_TEXT,
                "workspace_path": SQLITE_TEXT,
                "fk_patient_id": SQLITE_INTEGER,
                "fk_study_id": SQLITE_INTEGER,
            },
        ):
            valid_status = False

        conn.close()

        return valid_status

    def __get_tag_ids(self, aps_tags: list[str]) -> list[int]:
        aps_tag_ids: list[int] = []
        cur = self.__conn.cursor()
        for tag in aps_tags:
            cur.execute("SELECT id FROM series_tag WHERE name = ?;", (tag,))
            result_tuple = cur.fetchone()
            if result_tuple == None:
                continue
            aps_tag_ids.append(result_tuple[0])
        return aps_tag_ids

    def __is_series_have_tags(self, series_db_id: int, aps_tag_ids: list[int]) -> bool:
        cur = self.__conn.cursor()
        for tag_db_id in aps_tag_ids:
            cur.execute(
                "SELECT id FROM series_to_series_tag WHERE fk_series_id = ? AND fk_tag_id = ?;",
                (series_db_id, tag_db_id),
            )
            if cur.fetchone() == None:
                return False
        return True

    def __get_aps_series_db_ids(self, aps_tag_ids: list[int]) -> list[int]:
        series_db_ids: list[int] = []
        cur = self.__conn.cursor()
        cur.execute("SELECT id FROM series;")
        for (series_db_id,) in cur.fetchall():
            # 判断是否存在所有指定的aps标签
            if not self.__is_series_have_tags(series_db_id, aps_tag_ids):
                continue
            series_db_ids.append(series_db_id)
        return series_db_ids

    def __get_series_by_series_db_id(
        self, series_db_ids: list[int]
    ) -> list[tuple[int, int, str, str, int, int, int]]:
        cur = self.__conn.cursor()
        series_list: list[tuple[int, int, str, str, int, int, int]] = []

        for series_db_id in series_db_ids:
            cur.execute(
                "SELECT series_number, series_instance_uid, workspace_path, sop_instance_count, fk_patient_id, fk_study_id FROM series "
                "WHERE id = ? "
                "AND modality = 'CT';",
                (series_db_id,),
            )
            for (
                series_number,
                series_instance_uid,
                workspace_path,
                sop_instance_count,
                patient_db_id,
                study_db_id,
            ) in cur.fetchall():
                series_list.append(
                    (
                        series_db_id,
                        series_number,
                        series_instance_uid,
                        workspace_path,
                        patient_db_id,
                        study_db_id,
                        sop_instance_count,
                    )
                )
        return series_list

    def get_series_list_by_aps_tags(self, aps_tags: list[str]) -> SeriesList:
        """通过 aps 标签获取序列信息（这些序列都成功运行指定的 aps 并生成相应掩膜结果）

        Args:
            aps_tags (list[str]): IN aps 标签列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        assert aps_tags.__len__() != 0

        # 获取 aps 的序列标签ID
        aps_tag_ids = self.__get_tag_ids(aps_tags)
        if aps_tag_ids.__len__() == 0:
            return []

        # 遍历每个序列判断是否存在相应的标签
        series_db_ids = self.__get_aps_series_db_ids(aps_tag_ids)
        if series_db_ids.__len__() == 0:
            return []

        series_info_list: SeriesList = []
        # 获取序列的信息
        for (
            series_db_id,
            series_number,
            series_instance_uid,
            workspace_path,
            patient_db_id,
            study_db_id,
            sop_instance_count,
        ) in self.__get_series_by_series_db_id(series_db_ids):
            self.__append_series_info(
                patient_db_id,
                study_db_id,
                series_db_id,
                series_number,
                series_instance_uid,
                workspace_path,
                sop_instance_count,
                series_info_list,
            )

        return series_info_list

    def get_db_id_by_series_uid(self, series_uid: str) -> tuple[int, int] | None:
        """通过序列uid获取相应的检查数据库id和序列数据库id

        Args:
            series_uid (str): 序列uid

        Returns:
            tuple[int, int] | None: 检查数据库id，序列数据库id；序列不存在
        """
        assert series_uid.__len__() != 0
        cur = self.__conn.cursor()
        cur.execute(
            "SELECT fk_study_id, id FROM series WHERE series_instance_uid = ?",
            (series_uid,),
        )

        res = cur.fetchone()
        if res == None:
            return None
        return res

    def get_series_list_by_patient_id(self, patient_ids: list[str]) -> SeriesList:
        """通过 patient id 列表获取序列信息

        Args:
            patient_ids (str): patient id 列表

        Returns:
            SeriesList: 序列信息列表对象
        """
        assert patient_ids.__len__() != 0
        series_info_list: SeriesList = []

        # 获取相应的患者信息
        for (
            patient_db_id,
            patient_id,
            patient_name,
        ) in self.__get_patients_by_patient_id(patient_ids):
            # 获取患者对应的检查信息
            for (
                study_db_id,
                study_id,
                accession_number,
                study_instance_uid,
            ) in self.__get_studies_by_patient_db_id(patient_db_id):
                # 获取检查对应的序列信息
                self.__append_series_by_study_db_id(
                    series_info_list,
                    study_db_id,
                    patient_id,
                    patient_name,
                    study_id,
                    accession_number,
                    study_instance_uid,
                    updated_at
                )
        return series_info_list

    def __get_all_patients_by_patient_id(
            self, patient_ids: list[str]
        ) -> list[tuple[int, str, str]]:
            """根据患者ID列表获取相应患者信息

            Args:
                patient_ids (list[str]): 患者ID列表

            Returns:
                list[tuple[int, str, str]]: 患者信息 (patient_db_id, patient_id, patient_name)
            """
            cur = self.__conn.cursor()
            # patient = {}
            # for patient_id in patient_ids:
            #     # fmt: off
            #     cur.execute(
            #         f"SELECT id, patient_name, patient_sex, "
            #         "patient_birth_date, study_count, series_count, sop_instance_count," 
            #         "imported_at, legacy_site_key, created_at, updated_at FROM patient "
            #         "WHERE patient_id = ?;",
            #         (patient_id,),
            #     )
            #     # fmt on
            #     patient['patient'] = {}
            #     for (ids, patient_name, patient_sex, patient_birth_date, study_count, series_count,
            #     sop_instance_count, imported_at, legacy_site_key, created_at, updated_at) in cur.fetchall():
            #         patient['patient']["ids"] = ids
            #         patient['patient']["patient_name"] = patient_name
            #         patient['patient']["patient_sex"] = patient_sex
            #         patient['patient']["patient_birth_date"] = patient_birth_date
            #         patient['patient']["study_count"] = study_count
            #         patient['patient']["series_count"] = series_count
            #         patient['patient']["sop_instance_count"] = sop_instance_count
            #         patient['patient']["imported_at"] = imported_at
            #         patient['patient']["legacy_site_key"] = legacy_site_key
            #         patient['patient']["created_at"] = created_at
            #         patient['patient']["updated_at"] = updated_at

            cur.execute("SELECT * FROM schema_version;")
            rows = cur.fetchall()
            print("---------------1schema_version---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM sqlite_sequence;")
            rows = cur.fetchall()
            print("---------------2sqlite_sequence---------------")
            for row in rows:
                print(row)
                
            cur.execute("SELECT * FROM custom_column;")
            rows = cur.fetchall()
            print("---------------3custom_column---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM config;")
            rows = cur.fetchall()
            print("---------------4config---------------")
            for row in rows:
                print(row)
        
            cur.execute("SELECT * FROM deanonymizing_series;")
            rows = cur.fetchall()
            print("---------------5deanonymizing_series---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM deanonymizing_study;")
            rows = cur.fetchall()
            print("---------------6deanonymizing_study---------------")
            for row in rows:
                print(row)

            cur.execute("SELECT * FROM key_image;")
            rows = cur.fetchall()
            print("---------------7key_image---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM patient;")
            rows = cur.fetchall()
            print("---------------8patient---------------")
            for row in rows:
                print(row)

            cur.execute("SELECT * FROM patient_org;")
            rows = cur.fetchall()
            print("---------------9patient_org---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM series;")
            rows = cur.fetchall()
            print("---------------10series---------------")
            for row in rows:
                print(row)

            cur.execute("SELECT * FROM series_folder;")
            rows = cur.fetchall()
            print("---------------11series_folder---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM series_tag;")
            rows = cur.fetchall()
            print("---------------12series_tag---------------")
            for row in rows:
                print(row)
        
            cur.execute("SELECT * FROM series_to_series_tag;")
            rows = cur.fetchall()
            print("---------------13series_to_series_tag---------------")
            for row in rows:
                print(row)
            
            # cur.execute("SELECT * FROM sop_instance;")
            # rows = cur.fetchall()
            # print("---------------14sop_instance---------------")
            # for row in rows:
            #     print(row)

            cur.execute("SELECT * FROM study;")
            rows = cur.fetchall()
            print("---------------15study---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM study_comment;")
            rows = cur.fetchall()
            print("---------------16study_comment ---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM study_folder;")
            rows = cur.fetchall()
            print("---------------17study_folder---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM study_lock;")
            rows = cur.fetchall()
            print("---------------18study_lock---------------")
            for row in rows:
                print(row)

            cur.execute("SELECT * FROM study_report;")
            rows = cur.fetchall()
            print("---------------19study_report---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM study_result;")
            rows = cur.fetchall()
            print("---------------20study_result---------------")
            for row in rows:
                print(row)

            cur.execute("SELECT * FROM study_status;")
            rows = cur.fetchall()
            print("---------------21study_status---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM study_status_history;")
            rows = cur.fetchall()
            print("---------------22study_status_history---------------")
            for row in rows:
                print(row)
        
            cur.execute("SELECT * FROM study_tag;")
            rows = cur.fetchall()
            print("---------------23study_tag---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM study_to_study_tag ;")
            rows = cur.fetchall()
            print("---------------24study_to_study_tag ---------------")
            for row in rows:
                print(row)

            cur.execute("SELECT * FROM task;")
            rows = cur.fetchall()
            print("---------------25task---------------")
            for row in rows:
                print(row)
            
            cur.execute("SELECT * FROM volume_disk;")
            rows = cur.fetchall()
            print("---------------26volume_disk ---------------")
            for row in rows:
                print(row)


    def get_all_table_name(self):
        cur = self.__conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table = cur.fetchall()
        for i in table:
            print(i[0] + "\n")

        cur.execute("PRAGMA table_info('patient')")
        columns = cur.fetchall()
        print(columns)