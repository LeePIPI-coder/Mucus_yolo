from argparse import ArgumentParser, RawDescriptionHelpFormatter
from logging import DEBUG
from os import path

from .. import settings
from ..logger import logger
from ..StorageReader import check_storage_valid, get_storage_obj
from ..StorageReader.SeriesList import SeriesList
from .UploadMask import (
    upload_research_masks,
    upload_empty_masks,
    calling_aps_task,
)

__all__ = [
    "APSMaskConsoleAPP",
]


class APSMaskConsoleAPP:
    """控制台命令程序"""

    def __init__(self) -> None:
        """构造函数，添加控制台选项"""
        self.__parser = ArgumentParser(
            settings.PROJECT_NAME,
            description=settings.PROJECT_DESCRIPTION
            + "\n"
            + settings.PROJECT_COPYRIGHT_INFO,
            add_help=False,
            formatter_class=RawDescriptionHelpFormatter,
            usage="simAPSMaskUpload [选项] storage_dir",
        )
        self.__add_options()

    def __add_options(self):
        """添加控制台命令选项，仅供 __init__() 调用"""

        CONST_STORE = "store"
        CONST_APPEND = "append"
        CONST_STORE_TRUE = "store_true"

        # 基本选项
        self.__parser.add_argument(
            "-h", "--help", help="打印帮助信息并退出。", action="help"
        )
        self.__parser.add_argument(
            "-v",
            "--version",
            help="打印当前工具的版本。",
            action="version",
            version=settings.PROJECT_VERSION_TEXT,
        )
        # 可选参数
        self.__parser.add_argument(
            "-d", "--debug", help="启动调试。", action=CONST_STORE_TRUE
        )

        # 位置参数
        self.__parser.add_argument(
            "storage_dir",
            help="指定要进行掩膜上传处理的 AVIEW 存储库路径。不指定 --connect-str 时默认访问 sqlite 数据库。",
            action=CONST_STORE,
        )

        # 数据库连接选项
        self.__parser.add_argument(
            "--connect-str",
            help="指定存储库数据库的连接字符串，不使用时默认访问 sqlite 数据库。",
            action=CONST_STORE,
        )

        # 序列选择选项
        self.__parser.add_argument(
            "-p",
            "--patient-id",
            help="指定要进行掩膜上传处理的患者ID。",
            action=CONST_APPEND,
        )
        self.__parser.add_argument(
            "-s",
            "--study-id",
            help="指定要进行掩膜上传处理的检查ID。",
            action=CONST_APPEND,
        )
        self.__parser.add_argument(
            "--access-num",
            help="指定要进行掩膜指上传处理的 access number。",
            action=CONST_APPEND,
        )
        self.__parser.add_argument(
            "--study-uid",
            help="指定要进行掩膜上传处理的 study instance uid。",
            action=CONST_APPEND,
        )
        self.__parser.add_argument(
            "--series-uid",
            help="指定要进行掩膜上传处理的 series instance uid。",
            action=CONST_APPEND,
        )
        self.__parser.add_argument(
            "-a",
            "--all",
            help="对存储库中所有的序列进行掩膜上传处理。",
            action=CONST_STORE_TRUE,
        )

        #! TODO: 获取带有 aps 分析标签的序列
        self.__parser.add_argument(
            "--aps-tag",
            help="含有所有指定 aps 标签的序列进行掩膜上传",
            action=CONST_APPEND,
            choices=settings.APS_TAG,
        )

        self.__parser.add_argument(
            "--empty-type",
            help="指定上传的空白掩膜名称，并在注释处保留命名",
            action=CONST_APPEND,
        )

        # 执行类型选择
        self.__parser.add_argument(
            "-t",
            "--mask-type",
            help="指定要进行上传处理的掩膜类型。",
            action=CONST_APPEND,
            choices=settings.APPLY_MASK_TYPES,
        )

        # TODO:
        # 文件打印选项
        # self.__parser.add_argument(
        #     "-c",
        #     "--csv",
        #     help="将已经上传的序列以 csv 列表文件形式进行保存。",
        #     action=CONST_STORE_TRUE,
        # )

        # 调用AVIEW task接口时使用的地址
        self.__parser.add_argument(
            "--task-address",
            help="调用 AVIEW 进行影像组学计算的接口地址，如 127.0.0.1:5470 或 http://localhost:5470。默认不使用将不会进行调用影像组学计算的接口。",
            action=CONST_STORE,
        )

        self.__parser.add_argument(
            "--data-token",
            help="调用 AVIEW 进行影像组学计算接口时使用的 data_token。默认不使用将不会进行调用影像组学计算的接口。",
            action=CONST_STORE,
        )

    def exec(self, argv: list[str]) -> None:
        """控制台程序运行

        Args:
            argv (list[str]): 除启动程序外的运行参数
        """
        args = self.__parser.parse_args(argv)

        # args.debug            bool
        # args.storage_dir      str
        # args.connect_str      str
        # args.patient_id       list[str]
        # args.study_id         list[str]
        # args.access_num       list[str]
        # args.study_uid        list[str]
        # args.series_uid       list[str]
        # args.mask_type        list[str]
        # args.all              bool
        # args.csv              str
        # args.aps_tag          list[str]
        # args.empty_type       list[str]
        # args.task_address     str
        # args.data_token       str

        #! TODO: 上面的选项参数存在重复的情况也是需要处理的

        # 是否启动调试
        if args.debug != None and args.debug == True:
            logger.setLevel(DEBUG)

        logger.debug("simAPSMaskUpload 程序执行时使用的参数为 %s" % argv)

        # 检查存储库有效性
        if not path.exists(args.storage_dir):
            logger.error(
                "指定的 AVIEW 存储库路径 %s 不存在，请检查后重试！" % args.storage_dir
            )
            return

        if not check_storage_valid(args.storage_dir, args.connect_str):
            if args.connect_str == None or args.connect_str.__len__() == 0:
                logger.error(
                    "指定的 AVIEW 存储库路径 %s 无效，请检查后重试！" % args.storage_dir
                )
            else:
                logger.error(
                    "指定的 AVIEW 存储库路径 %s 和连接字符串 %s 无效，请检查后重试！"
                    % (args.storage_dir, args.connect_str)
                )
            return

        # 获取所有需要掩膜上传处理的序列
        storage_obj = get_storage_obj(args.storage_dir, args.connect_str)
        series_list = SeriesList()
        if args.all:
            series_list = storage_obj.get_all_series_list()
        elif args.aps_tag != None and args.aps_tag.__len__() != 0:
            series_list += storage_obj.get_series_list_by_aps_tags(args.aps_tag)
        else:
            if args.patient_id != None and args.patient_id.__len__() != 0:
                series_list += storage_obj.get_series_list_by_patient_ids(
                    args.patient_id
                )
            if args.study_id != None and args.study_id.__len__() != 0:
                series_list += storage_obj.get_series_list_by_study_ids(args.study_id)
            if args.access_num != None and args.access_num.__len__() != 0:
                series_list += storage_obj.get_series_list_by_access_nums(
                    args.access_num
                )
            if args.study_uid != None and args.study_uid.__len__() != 0:
                series_list += storage_obj.get_series_list_by_study_uids(args.study_uid)
            if args.series_uid != None and args.series_uid.__len__() != 0:
                series_list += storage_obj.get_series_list_by_series_uids(
                    args.series_uid
                )
        logger.info("从数据库中获取到待掩膜上传的序列数为 %d" % series_list.__len__())
        if logger.isEnabledFor(DEBUG):
            for i, series_info in enumerate(series_list):
                logger.debug(
                    "%-4d %s, %s, %s, %s, %s, %d, %s, %s, %s, %s, %d, %d, %d"
                    % (
                        i,
                        series_info.patient_id,
                        series_info.patient_name,
                        series_info.study_id,
                        series_info.access_number,
                        series_info.study_instance_uid,
                        series_info.series_number,
                        series_info.series_instance_uid,
                        series_info.abs_path,
                        series_info.patient_pos_max,
                        series_info.patient_pos_min,
                        series_info.rows,
                        series_info.cols,
                        series_info.sop_instance_count,
                    )
                )

        # 对所有的序列进行掩膜的上传，同时记录上传情况
        if series_list.__len__() == 0:
            logger.info("未指定参数或通过参数解析得到的序列数为 0，退出上传。")
            return

        upload_results = None
        if args.mask_type != None and args.mask_type.__len__() != 0:
            logger.debug("上传aps分析结果掩膜数据类型 %s " % args.mask_type)
            upload_results = upload_research_masks(series_list, args.mask_type)
            logger.info(
                "已完成上传 %d 例序列aps掩膜数据，上传的掩膜类型列表为 %s"
                % (upload_results.__len__(), args.mask_type),
            )
        else:
            if args.empty_type == None or args.empty_type.__len__() == 0:
                logger.info("未指定的待上传的掩膜类型，程序退出！")
                return

        if args.empty_type != None and args.empty_type.__len__() != 0:
            logger.debug("上传空掩膜名称列表 %s ", args.empty_type)
            upload_empty_masks(series_list, args.empty_type)

        # 是否调用接口进行组学计算
        if (
            args.task_address != None
            and args.task_address.__len__() != 0
            and args.data_token != None
            and args.data_token.__len__() != 0
            and upload_results != None
        ):
            calling_aps_task(args.task_address, args.data_token, upload_results, storage_obj)
        else:
            logger.debug("未指定调用AVIEW影像组学计算的接口地址或 data_token，不进行影像组学计算。")

        # TODO:
        # 是否进行 csv 文件的生成
        # if args.csv:
        #     save_csv(upload_results, csv_file)
        #     logger.info(
        #         "已将 %d 条序列掩膜上传结果保存到文件 %s 中。"
        #         % (upload_results.__len__(), args.csv)
        #     )