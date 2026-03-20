from os import path, mkdir
from pathlib import Path
from zipfile import ZipFile
from xml.etree.ElementTree import parse
from json import load, loads, dump, dumps
from typing import Any
from shutil import copy2
from datetime import datetime
# from lz4.frame import decompress
from gzip import compress
from re import match
from urllib import request, error

from ..settings import (
    MASK_ALL,
    MASK_WHOLE_LUNG,
    MASK_TYPES,
    MASK_LEFT_LUNG,
    MASK_RIGHT_LUNG,
    ACCESS_TOKEN,
    CALLING_COUNT,
)
from ..StorageReader.SeriesList import SeriesList, SeriesItem
from .UploadResult import UploadResult, UploadResults
from ..StorageReader.StorageReaderBase import StorageReaderBase

__all__ = ["upload_research_masks", "upload_empty_masks", "calling_aps_task"]

# lesinAnnot3D.json 文件字段
JSON_KEY_VERSION = "formatVersion"
JSON_VALUE_VERSION = 5
JSON_KEY_STUDY = "study"
JSON_KEY_STUDY_UID = "studyInstanceUid"
JSON_KEY_SERIES = "series"
JSON_KEY_SERIES_UID = "seriesInstanceUid"
JSON_KEY_LESION = "lesion"

xml_file = "data.xml"

# 给定字符串，将其中series、stduy参数传给json文件
def __generate_init_mask_json(series_info: SeriesItem) -> dict[str, Any]:
    series_dict: dict[str, Any] = {}
    series_dict[JSON_KEY_LESION] = []
    series_dict[JSON_KEY_SERIES_UID] = series_info.series_instance_uid
    json_series: list[dict[str, Any]] = []
    json_series.append(series_dict)

    study_dict: dict[str, Any] = {}
    study_dict[JSON_KEY_SERIES] = json_series
    study_dict[JSON_KEY_STUDY_UID] = series_info.study_instance_uid
    json_study: list[dict[str, Any]] = []
    json_study.append(study_dict)

    json_obj: dict[str, Any] = {}
    json_obj[JSON_KEY_VERSION] = 5
    json_obj[JSON_KEY_STUDY] = json_study

    return json_obj


def __log_error_key(
    series_info: SeriesItem, json_file: str, key_name: str, err_msg: str
) -> None:
    logger.error(
        "检查ID %s 对应序列 %d 的 research3D 文件 %s 中字段 %s 无效。%s"
        % (
            series_info.study_id,
            series_info.series_number,
            json_file,
            key_name,
            err_msg,
        )
    )


def __check_key_exist(
    series_info: SeriesItem, json_file: str, key_name: str, json_dict: dict[str, Any]
) -> bool:
    if key_name not in json_dict.keys():
        __log_error_key(series_info, json_file, key_name, "字段不存在")
        return False
    return True


def __check_json_field(
    series_info: SeriesItem,
    json_file: str,
    json_dict: dict[str, Any],
    json_key_name: str,
    json_key_uid: str,
    value_uid: str,
) -> bool:
    if not __check_key_exist(series_info, json_file, json_key_name, json_dict):
        return False

    json_list: list[dict[str, Any]] = json_dict[json_key_name]
    if type(json_list) != list or json_list.__len__() == 0:
        __log_error_key(series_info, json_file, json_key_name, "不是列表或列表长度为 0")
        return False

    json_sub_dict: dict[str, Any] = json_list[0]
    if type(json_sub_dict) != dict:
        __log_error_key(series_info, json_file, json_key_name, "值类型不是字典")
        return False

    if not __check_key_exist(series_info, json_file, json_key_uid, json_sub_dict):
        return False

    json_uid: str = json_sub_dict[json_key_uid]
    if json_uid != value_uid:
        __log_error_key(
            series_info,
            json_file,
            json_key_uid,
            "UID不一致，JSON为 %s, 数据库为 %s"
            % (json_uid, series_info.study_instance_uid),
        )
        return False

    return True


def __check_existed_json_valid(
    lesion_annot_3D_json: dict[str, Any], series_info: SeriesItem, json_file: str
) -> bool:
    # root
    json_keys = lesion_annot_3D_json.keys()
    if (
        JSON_KEY_VERSION not in json_keys
        or lesion_annot_3D_json[JSON_KEY_VERSION] != JSON_VALUE_VERSION
    ):
        __log_error_key(
            series_info,
            json_file,
            JSON_KEY_VERSION,
            "字段不存在或值不等于 %d" % JSON_VALUE_VERSION,
        )
        return False

    # study 字段
    if not __check_json_field(
        series_info,
        json_file,
        lesion_annot_3D_json,
        JSON_KEY_STUDY,
        JSON_KEY_STUDY_UID,
        series_info.study_instance_uid,
    ):
        return False

    json_study_dict = lesion_annot_3D_json[JSON_KEY_STUDY][0]
    # series 字段
    if not __check_json_field(
        series_info,
        json_file,
        json_study_dict,
        JSON_KEY_SERIES,
        JSON_KEY_SERIES_UID,
        series_info.series_instance_uid,
    ):
        return False

    json_series_dict = json_study_dict[JSON_KEY_SERIES][0]
    if not __check_key_exist(series_info, json_file, JSON_KEY_LESION, json_series_dict):
        # 如果没有 lesion 节点，追加一个 lesion 节点
        json_series_dict[JSON_KEY_LESION] = []
        logger.info(
            "检查ID %s 对应序列 %d 的 research3D 文件 %s 中不存在字段 %s，程序中进行追加。上一条 ERROR 可以忽略"
            % (
                series_info.study_id,
                series_info.series_number,
                json_file,
                JSON_KEY_LESION,
            )
        )
    else:
        # 可以为空但必须是列表类型
        if type(json_series_dict[JSON_KEY_LESION]) != list:
            __log_error_key(series_info, json_file, JSON_KEY_LESION, "不是列表类型")
            return False

    return True


def __backup_json_file(json_file: str) -> str:
    json_backup_file = (
        Path(json_file).cwd().__str__()
        + "/lesionAnnot3D_backup_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    copy2(json_file, json_backup_file)
    return json_backup_file


def __get_data_from_json_file(
    json_file: str, series_info: SeriesItem
) -> tuple[dict[str, Any], int]:
    with open(json_file, encoding="utf-8") as f:
        lesion_annot_3D_json = load(f)
        if not __check_existed_json_valid(lesion_annot_3D_json, series_info, json_file):
            backup_file = __backup_json_file(json_file)
            logger.error(
                "检查ID %s 对应序列 %d 的 research3D 文件 %s 中已经存在的内容无效，已备份到 %s。将会生成新的结果文件。"
                % (
                    series_info.study_id,
                    series_info.series_number,
                    json_file,
                    backup_file,
                )
            )
            lesion_annot_3D_json = __generate_init_mask_json(series_info)
            start_lesion_index = 0
        else:
            start_lesion_index = lesion_annot_3D_json[JSON_KEY_STUDY][0][JSON_KEY_SERIES][0][JSON_KEY_LESION].__len__()
            logger.info(
                f"""检查ID {series_info.study_id} 对应序列 {series_info.series_number} 的 research3D 文件 
                        {json_file} 中已经存在 {start_lesion_index} 个病变, 新的掩膜病变将会追加。""")

    return lesion_annot_3D_json, start_lesion_index


def __get_xml_data(ziped_file: ZipFile, series_info: SeriesItem, mask_zip: Path) -> str:
    with ziped_file.open(xml_file) as xml_data:
        # 从 xml 文件中获取掩膜信息
        # 尺寸信息校验
        xml_tree = parse(xml_data)
        xml_X = xml_tree.find(
            "PlainData/storage/data_objitem/objItem/object_item/segMask/segmask/DimX"
        )
        xml_Y = xml_tree.find(
            "PlainData/storage/data_objitem/objItem/object_item/segMask/segmask/DimY"
        )
        xml_Z = xml_tree.find(
            "PlainData/storage/data_objitem/objItem/object_item/segMask/segmask/DimZ"
        )
        # fmt: off
        if (
            xml_X == None or xml_X.text == None or xml_X.text.__len__() == 0 or
            xml_Y == None or xml_Y.text == None or xml_Y.text.__len__() == 0 or
            xml_Z == None or xml_Z.text == None or xml_Z.text.__len__() == 0
        ):
        # fmt: on
            raise KeyError(
                "检查ID %s 对应序列 %d 的掩膜文件 %s 的 %s 文件没有 DimX 或 DimY 或 DimZ 字段"
                % (
                    series_info.study_id,
                    series_info.series_number,
                    ziped_file,
                    xml_data,
                )
            )

        dim_X = int(xml_X.text)
        dim_Y = int(xml_Y.text)
        dim_Z = int(xml_Z.text)
        if (
            series_info.rows != dim_X
            or series_info.cols != dim_Y
            or series_info.sop_instance_count != dim_Z
        ):
            # fmt: off
            raise RuntimeError(
                "检查ID %s 对应序列 %d 的掩膜文件 %s 中的维度(%d, %d, %d)与数据库中记录的维度(%d, %d, %d)不一致"
                % (
                    series_info.study_id, series_info.series_number, mask_zip,
                    dim_X, dim_Y, dim_Z,
                    series_info.rows,series_info.cols, series_info.sop_instance_count,
                )
            )
            # fmt: on

        # 掩膜文件信息
        xml_mask_file = xml_tree.find("BinaryData/storage/data_objitem/objItem/object_item/segMask/segmask/maskData")
        if xml_mask_file == None or xml_mask_file.text == None or xml_mask_file.text.__len__() == 0:
            raise KeyError(
                "检查ID %s 对应序列 %d 的掩膜文件 %s 的 %s 文件没有 maskData 字段"
                % (
                    series_info.study_id,
                    series_info.series_number,
                    ziped_file,
                    xml_data,
                )
            )

    return xml_mask_file.text


def __generate_json_lesion(
    series_info: SeriesItem, mask_file: str, has_pixel_mask: bool, mask_name: str, detect_box: list
) -> dict[str, Any]:
    json_lesion: dict[str, Any] = {}
    json_lesion["hasPixelMask"] = has_pixel_mask
    json_lesion["lesionType"] = "ELesionAnnotType_ROI_3D"
    json_lesion["maskFileNameSegmask"] = mask_file
    # json_lesion["maskPos"] = f"{series_info.top_left}-{series_info.bottom_right}"
    # json_lesion["measureString"] = f"{series_info.patient_vol} cc"
    json_lesion["roi_patientPos_max"] = detect_box[1]
    json_lesion["roi_patientPos_min"] = detect_box[0]
    json_lesion["sopInstanceUid"] = series_info.sopInstanceUid.__str__()
    json_comment_dict = {}
    json_comment_dict["__valid"] = True
    json_comment_dict["annotation"] = mask_name
    json_lesion["userAnnotComment"] = json_comment_dict
    return json_lesion


def __generate_lesion_file_and_append_lesion(
    dim_X: int,
    dim_Y: int,
    dim_Z: int,
    gz_compressed_mask: bytes | bytearray,
    series_info: SeriesItem,
    results_dir: str,
    mask_name: str,
    json_lesions: list[dict[str, Any]],
    start_lesion_index: int,
    detect_bbox: dict[str, Any],
) -> None:

    lesion_stem = "lesionAnnot3D-%03d.%dx%dx%d" % (
        start_lesion_index,
        dim_X,
        dim_Y,
        dim_Z,
    )
    lesion_temp = results_dir + "/" + lesion_stem
    lesion_suffix = ".psegmaskz"
    lesion_file = lesion_temp + lesion_suffix
    if path.isfile(lesion_file):
        backup_file = (
            lesion_temp
            + "_backup_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + lesion_suffix
        )
        copy2(lesion_file, backup_file)
        logger.warning(
            "检查ID %s 对应序列 %d 已存在research掩膜文件 %s，现备份至 %s"
            % (
                series_info.study_id,
                series_info.series_number,
                lesion_file,
                backup_file,
            )
        )
    with open(lesion_file, "wb") as f:
        f.write(gz_compressed_mask)

    # 追加病变到 lesionAnnot3D.json 文件中
    for i in range(len(detect_bbox)):
        detect_box = detect_bbox[f'{i}']
        json_lesions.append(
            __generate_json_lesion(
                series_info, lesion_stem + lesion_suffix, True, mask_name, detect_box
            )
        )


def __upload_mask(
    series_info: SeriesItem,
    mask_zip: Path,
    results_dir: str,
    json_lesions: list[dict[str, Any]],
    start_lesion_index: int,
) -> None:
    ziped_file = ZipFile(mask_zip)
    maskData_file = __get_xml_data(ziped_file, series_info, mask_zip)

    # 读取lz4压缩的 aps 掩膜文件，转存为gzip压缩的 research 掩膜文件
    with ziped_file.open(maskData_file) as lz4_data:
        decompressed_mask: bytes = decompress(lz4_data.read())
        gz_compressed_mask = compress(decompressed_mask)

    # 生成 lesion 文件并添加 lesion 信息到 lesionAnnot3D.json 对象中
    __generate_lesion_file_and_append_lesion(
        series_info.rows,
        series_info.cols,
        series_info.sop_instance_count,
        gz_compressed_mask,
        series_info,
        results_dir,
        mask_zip.stem,
        json_lesions,
        start_lesion_index,
    )


def __upload_mask_whole_lung(
    series_info: SeriesItem,
    left_lung_zip: Path,
    right_lung_zip: Path,
    results_dir: str,
    json_lesions: list[dict[str, Any]],
    start_lesion_index: int,
):
    # 读左肺
    ziped_LL_file = ZipFile(left_lung_zip)
    maskData_file_LL = __get_xml_data(ziped_LL_file, series_info, left_lung_zip)

    # 读右肺
    ziped_RL_file = ZipFile(right_lung_zip)
    maskData_file_RL = __get_xml_data(ziped_RL_file, series_info, left_lung_zip)

    # 叠加掩膜
    with ziped_LL_file.open(maskData_file_LL) as lz4_data_LL:
        decompressed_mask_LL: bytes = decompress(lz4_data_LL.read())
    with ziped_RL_file.open(maskData_file_RL) as lz4_data_RL:
        decompressed_mask_RL: bytes = decompress(lz4_data_RL.read())

    mask_len: int = decompressed_mask_LL.__len__()
    if mask_len != decompressed_mask_RL.__len__():
        raise RuntimeError(
            "检查ID %s 对应序列 %d 的左肺掩膜 %s 与右肺掩膜 %s 解压后的数据长度不一致"
            % (
                series_info.study_id,
                series_info.series_number,
                left_lung_zip,
                right_lung_zip,
            )
        )
    mask_whole_lung = bytearray(mask_len)
    for i in range(mask_len):
        mask_whole_lung[i] = decompressed_mask_LL[i] | decompressed_mask_RL[i]

    # 生成 lesion 文件并添加 lesion 信息到 lesionAnnot3D.json 对象中
    __generate_lesion_file_and_append_lesion(
        series_info.rows,
        series_info.cols,
        series_info.sop_instance_count,
        mask_whole_lung,
        series_info,
        results_dir,
        MASK_WHOLE_LUNG,
        json_lesions,
        start_lesion_index,
    )


def __generate_annot_json_if_not_existed(
    results_dir: str, lesion_annot_3D_file: str, series_info: SeriesItem
) -> tuple[dict[str, Any], int]:
    if not path.isdir(results_dir):
        mkdir(results_dir)
        logger.debug(
            "检查ID %s 对应序列 %d 的结果文件目录 %s 不存在"
            % (
                series_info.study_id,
                series_info.series_number,
                results_dir,
            )
        )
        return __generate_init_mask_json(series_info), 0
    else:
        # 假如已经存在 lesionAnnot3D.json，读取其中内容
        if not path.isfile(lesion_annot_3D_file):
            logger.debug(
                "检查ID %s 对应序列 %d 的 research3D 文件 %s 不存在"
                % (
                    series_info.study_id,
                    series_info.series_number,
                    lesion_annot_3D_file,
                )
            )
            return __generate_init_mask_json(series_info), 0
        else:
            # 尝试在现有 lesionAnnot3D.json 中追加新的病变
            return __get_data_from_json_file(lesion_annot_3D_file, series_info)


def __upload_single_series_masks(
    series_info: SeriesItem, objects_dir: str, mask_types: list[str]
) -> list[str]:
    results_dir = series_info.abs_path + "/stor/results"
    lesion_annot_3D_file = results_dir + "/lesionAnnot3D.json"
    lesion_annot_3D_json, start_lesion_index = __generate_annot_json_if_not_existed(
        results_dir, lesion_annot_3D_file, series_info
    )

    json_lesion_list = lesion_annot_3D_json[JSON_KEY_STUDY][0][JSON_KEY_SERIES][0][
        JSON_KEY_LESION
    ]
    uploaded_mask_types: list[str] = []
    left_lung_zip = None
    right_lung_zip = None
    # 上传序列路径下的掩膜
    for mask_zip in Path(objects_dir).rglob("*.zip"):
        mask_type = mask_zip.stem

        # 判断左右肺掩膜文件是否存在，用于后续生成
        if left_lung_zip == None and mask_type == MASK_LEFT_LUNG:
            left_lung_zip = mask_zip
        if right_lung_zip == None and mask_type == MASK_RIGHT_LUNG:
            right_lung_zip = mask_zip

        if mask_type not in mask_types:
            continue

        # 上传其他路径下有的掩膜
        __upload_mask(
            series_info,
            mask_zip,
            results_dir,
            json_lesion_list,
            start_lesion_index,
        )
        uploaded_mask_types.append(mask_type)
        start_lesion_index += 1
        logger.debug(
            "成功加载检查ID %s 对应序列 %d 的序列路径 %s 下的掩膜 %s"
            % (
                series_info.study_id,
                series_info.series_number,
                series_info.abs_path,
                mask_type,
            )
        )

    # 请求上传全肺
    if MASK_WHOLE_LUNG in mask_types:
        # 左右肺掩膜文件存在
        if left_lung_zip == None or right_lung_zip == None:
            logger.warning(
                "想要上传检查ID %s 对应序列 %d 的全肺掩膜，但左肺掩膜或右肺掩膜文件不存在，无法进行上传"
                % (series_info.study_id, series_info.series_number)
            )
        else:
            # 上传全肺掩膜
            __upload_mask_whole_lung(
                series_info,
                left_lung_zip,
                right_lung_zip,
                results_dir,
                json_lesion_list,
                start_lesion_index,
            )
            uploaded_mask_types.append(MASK_WHOLE_LUNG)
            logger.debug(
                "成功加载检查ID %s 对应序列 %d 的全肺掩膜"
                % (series_info.study_id, series_info.series_number)
            )

    # 生成 lesionAnnot3D.json 数据文件
    if uploaded_mask_types.__len__() != 0:
        with open(lesion_annot_3D_file, "w", encoding="utf-8") as f:
            dump(lesion_annot_3D_json, f)

    return uploaded_mask_types


def __is_series_path_existed(series_info: SeriesItem) -> bool:
    if not path.isdir(series_info.abs_path):
        logger.error(
            "检查ID %s 对应序列 %d 的序列路径 %s 不存在"
            % (
                series_info.study_id,
                series_info.series_number,
                series_info.abs_path,
            )
        )
        return False
    return True


def __upload_series_masks(
    series_list: SeriesList, mask_types: list[str]
) -> UploadResults:
    assert mask_types.__len__() != 0

    upload_results: UploadResults = []
    for series_info in series_list:
        uploaded_mask_types: list[str] = []
        # 检查路径有效性
        if not __is_series_path_existed(series_info):
            continue

        objects_dir = series_info.abs_path + "/stor/objects"
        if not path.isdir(objects_dir):
            logger.debug(
                "检查ID %s 对应序列 %d 的 objects 掩膜路径 %s 不存在"
                % (series_info.study_id, series_info.series_number, objects_dir)
            )
            continue

        try:
            uploaded_mask_types = __upload_single_series_masks(
                series_info, objects_dir, mask_types
            )
        except Exception as e:
            logger.error(
                "检查ID %s 对应序列 %d 的掩膜上传过程中出现异常，异常原因为 %s"
                % (series_info.study_id, series_info.series_number, e)
            )
            continue

        upload_results.append(UploadResult(series_info, uploaded_mask_types))

    return upload_results


def upload_research_masks(
    series_list: SeriesList, mask_types: list[str]
) -> UploadResults:
    """上传序列的aps掩膜结果

    Args:
        series_list (SeriesList): 序列信息列表
        mask_types (list[str]): 要上传的掩膜类型列表

    Returns:
        UploadResults: 上传情况
    """
    assert series_list.__len__() != 0
    assert mask_types.__len__() != 0

    if MASK_ALL in mask_types:
        # 上传所有掩膜
        return __upload_series_masks(series_list, MASK_TYPES)
    else:
        # 上传指定掩膜
        # TODO: wholeLung 不成功
        return __upload_series_masks(series_list, mask_types)


def upload_empty_masks(series_list: SeriesList, mask_names: list[str]) -> None:
    """上传空掩膜，在注释字段添加指定的掩膜名称

    Args:
        series_list (SeriesList): 序列信息列表
        type_names (list[str]): 空掩膜名称列表
    """
    assert series_list.__len__() != 0
    assert mask_names.__len__() != 0

    for series_info in series_list:
        # results 文件夹，lesionAnnot3D.json 文件是否存在，不存在就添加
        if not __is_series_path_existed(series_info):
            continue
        results_dir = series_info.abs_path + "/stor/results"
        lesion_annot_3D_file = results_dir + "/lesionAnnot3D.json"
        lesion_annot_3D_json, start_lesion_index = __generate_annot_json_if_not_existed(  # type: ignore
            results_dir, lesion_annot_3D_file, series_info
        )

        json_lesion_list = lesion_annot_3D_json[JSON_KEY_STUDY][0][JSON_KEY_SERIES][0][
            JSON_KEY_LESION
        ]

        # 追加空掩膜
        for mask_name in mask_names:
            json_lesion_list.append(
                __generate_json_lesion(series_info, "", False, mask_name)
            )
            logger.debug(
                "检查ID %s 对应序列 %d 的research掩膜数据对象添加空掩膜病变 %s"
                % (series_info.study_id, series_info.series_number, mask_name)
            )

        # 保存 lesionAnnot3D.json 文件
        try:
            with open(lesion_annot_3D_file, "w", encoding="utf-8") as f:
                dump(lesion_annot_3D_json, f)
        except Exception as e:
            logger.error(
                "检查ID %s 对应序列 %d 的结果文件 lesionAnnot3D.json 无法成功保存。原因为 %s，内容为 %s"
                % (
                    series_info.study_id,
                    series_info.series_number,
                    e,
                    lesion_annot_3D_json,
                )
            )
            continue


def __update_access_token(req: request.Request, url_access_token: str) -> str | None:
    resp = request.urlopen(req)
    if resp.status != 200:
        logger.error(
            "无法调用指定地址的接口 %s，返回的状态码为 %d, 原因为 %s。无法调用影像组学计算"
            % (
                url_access_token,
                resp.status,
                resp.reason,
            )
        )
        return None
    access_token = loads(resp.read().decode("utf-8"))["access_token"]
    logger.debug(
        "调用接口 %s 返回的 access_token 为 %s" % (url_access_token, access_token)
    )
    return access_token


def calling_aps_task(
    address: str,
    data_token: str,
    upload_results: UploadResults,
    storage_obj: StorageReaderBase,
) -> None:
    """调用AVIEW影像组学计算任务接口

    Args:
        address (str): 接口地址 host:port
        upload_results (UploadResults): 成功进行掩膜上传的序列信息列表
        data_token (str): AVIEW 存储库的token，用于拼接调用url
        storage_obj (StorageReaderBase): 存储库对象
    """
    # 判断接口地址是否有效
    if match(r"^http(s)?://([\w\.-]+(:\d+)?)$", address) == None:
        if match(r"^(\d+\.){3}\d+:\d+$", address) == None:
            logger.error(
                "指定的接口调用地址 %s 不符合格式。无法调用影像组学计算" % address
            )
            return
        else:
            address = "http://" + address

    # 由于AVIEW接口调用方式的改变，每次 access-token 仅能维持5分钟，默认考虑每 100 次进行 token 的刷新
    init_count = 0
    # 首次纪录获取的 access-token
    req_data = {"access_token": ACCESS_TOKEN}
    url_access_token = address + "/api/v1/auth/authenticate/access-token"
    req = request.Request(
        url_access_token, dumps(req_data).encode("utf-8"), method="POST"
    )
    access_token = __update_access_token(req, url_access_token)
    if access_token == None:
        return

    # 对每个序列调用新建任务
    for result in upload_results:
        series_info = result.series_info
        # 获取检查数据库ID和序列数据库ID
        res_db_ids = storage_obj.get_db_id_by_series_uid(
            series_info.series_instance_uid
        )
        if res_db_ids == None:
            logger.error(
                "检查ID %s 的序列号 %d 查询检查数据库ID和序列数据库ID失败，序列号不存在"
                % (series_info.study_id, series_info.series_number)
            )
            continue

        study_db_id, series_db_id = res_db_ids

        # 拼接接口调用参数
        req_data = {
            "fk_study_id": study_db_id,
            "fk_series_ids": [series_db_id],
            "proc_name": "apsUpdateLesionsInfo",
            "proc_params": '[{"name":"update-2dview-lesions-info","type":"bool","value":"false"},{"name":"update-3dview-lesions-info","type":"bool","value":"true"},{"name":"capture-2dview-rois","type":"bool","value":"false"},{"name":"dcm-captured-imgs","type":"bool","value":"false"},{"name":"update-db","type":"bool","value":"true"}]',
            "dry_run": False,
        }
        url_task = address + "/api/v1/data/" + data_token + "/aps/tasks"
        req = request.Request(url_task, dumps(req_data).encode("utf-8"), method="POST")
        req.add_header("Authorization", "Bearer " + access_token)  # type: ignore

        # 调用 /api/v1/data/{data_token}/aps/tasks
        try:

            request.urlopen(req)
        except error.URLError as e:
            logger.error(
                "检查ID %s 的序列号 %d 调用影像组学计算接口 %s 失败，原因为 %s"
                % (series_info.study_id, series_info.series_number, url_task, e)
            )
            continue

        logger.debug(
            "检查ID %s 的序列号 %d 成功调用影响组学计算接口 %s"
            % (series_info.study_id, series_info.series_number, url_task)
        )
        # 是否达到次数要求，更新 access-token
        if init_count >= CALLING_COUNT:
            logger.debug("经过 %d 次请求后，更新新的 access-token" % (CALLING_COUNT,))
            access_token = __update_access_token(req, url_access_token)