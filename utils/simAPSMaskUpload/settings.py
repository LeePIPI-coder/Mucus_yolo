from datetime import date
from sys import argv
from os import path

__all__ = [
    "PROJECT_NAME",
    "PROJECT_VERSION_MAJOR",
    "PROJECT_VERSION_MINOR",
    "PROJECT_VERSION_PATCH",
    "PROJECT_VERSION_TEXT",
    "PACKING_DATE",
    "PROJECT_PACKING_DATA_8CH",
    "PROJECT_PACKING_DATA_STD",
    "PROJECT_COMPANY_NAME",
    "PROJECT_COPYRIGHT_INFO",
    "PROJECT_DESCRIPTION",
    "MASK_ALL",
    "APPLY_MASK_TYPES",
    "MASK_TYPES",
    "BASE_DIR",
    "DEFAULT_LOGGER_CONFIG_FILE",
    "AVIEW_SQLITE_KEY",
    "AVIEW_SQLITE_CIPHER",
    "AVIEW_SQLITE_KDF_ITER",
    "AVIEW_SQLITE_CIPHER_PAGE_SIZE",
    "CALLING_COUNT",
    "ACCESS_TOKEN",
]


##############################################################################
#  project info
##############################################################################
PROJECT_NAME = "simAPSMaskUpload"

# project version
PROJECT_VERSION_MAJOR: int = 1
PROJECT_VERSION_MINOR: int = 0
PROJECT_VERSION_PATCH: int = 1
# PROJECT_VERSION_PRERELEASE: str

if "PROJECT_VERSION_PRERELEASE" in vars():
    PROJECT_VERSION_TEXT: str = f"{PROJECT_VERSION_MAJOR}.{PROJECT_VERSION_MINOR}.{PROJECT_VERSION_PATCH}-{PROJECT_VERSION_PRERELEASE}"  # type: ignore
else:
    PROJECT_VERSION_TEXT: str = (  # type: ignore
        f"{PROJECT_VERSION_MAJOR}.{PROJECT_VERSION_MINOR}.{PROJECT_VERSION_PATCH}"
    )

# project packing date
PROJECT_PACKING_YEAR: int = 2024
PROJECT_PACKING_MON: int = 11
PROJECT_PACKING_DAY: int = 22

PACKING_DATE = date(PROJECT_PACKING_YEAR, PROJECT_PACKING_MON, PROJECT_PACKING_DAY)
PROJECT_PACKING_DATA_8CH: str = PACKING_DATE.strftime("%y%m%d")
PROJECT_PACKING_DATA_STD: str = PACKING_DATE.isoformat()

# company & copyright
PROJECT_COMPANY_NAME: str = "Hangzhou Smart Intelligent Technology Co., Ltd."
PROJECT_COPYRIGHT_INFO: str = (
    f"© 2019-{PROJECT_PACKING_YEAR} SIM Soft. All rights are reserved."
)

PROJECT_DESCRIPTION: str = (
    f"{PROJECT_NAME} 是用于上传 APS 分析后的结果掩膜到 research 页面中自动进行影像组学计算的工具。"
)

##############################################################################
# project settings
##############################################################################
# 要上传处理的掩膜类型
# fmt: off
MASK_ALL = "all"
MASK_WHOLE_LUNG = "LungSeg.Obj.WholeLung"

MASK_LEFT_LUNG = "LungSeg.Obj.LtLung"
MASK_RIGHT_LUNG = "LungSeg.Obj.RtLung"

APPLY_MASK_TYPES = [
    MASK_ALL,                                  # 一个特殊的掩膜选项

    "AirTrap.Obj.AirTrap",
    "AirTrap.Obj.Emphysema",
    "AirTrap.Obj.fAT",
    "AirTrap.Obj.Normal",

    "AirwaySeg.Obj.Airway",

    "CAC.Obj.Cal_LAD",
    "CAC.Obj.Cal_LCX",
    "CAC.Obj.Cal_LM",
    "CAC.Obj.Cal_RCA",

    "HAA.Obj.AeratedLungParenchyma_WOV",
    "HAA.Obj.AeratedLungParenchyma_WV",
    "HAA.Obj.BloodThrombus_WOV",
    "HAA.Obj.BloodThrombus_WV",
    "HAA.Obj.Emphysema_WOV",
    "HAA.Obj.Emphysema_WV",
    "HAA.Obj.GGODAD_WOV",
    "HAA.Obj.GGODAD_WV",
    "HAA.Obj.Lipids_WOV",
    "HAA.Obj.Lipids_WV",
    "HAA.Obj.LipoidsFluids_WOV",
    "HAA.Obj.LipoidsFluids_WV",
    "HAA.Obj.TransudateExudate_WOV",
    "HAA.Obj.TransudateExudate_WV",
    "LAA.Obj.Haa",
    "LAA.Obj.Laa",
    "LAASize.Obj.Size1",
    "LAASize.Obj.Size2",
    "LAASize.Obj.Size3",
    "LAASize.Obj.Size4",

    "LobeSeg.Obj.LtLower",
    "LobeSeg.Obj.LtUpper",
    "LobeSeg.Obj.RtLower",
    "LobeSeg.Obj.RtMiddle",
    "LobeSeg.Obj.RtUpper",

    "LungSeg.Obj.Airway",
    MASK_LEFT_LUNG,
    MASK_RIGHT_LUNG,

    MASK_WHOLE_LUNG,

    "LungTexture.Obj.Consolidation",
    "LungTexture.Obj.Emphysema",
    "LungTexture.Obj.GGO",
    "LungTexture.Obj.Honeycomb",
    "LungTexture.Obj.Normal",
    "LungTexture.Obj.Reticular",

    "LungVes.Obj.SingleVessel",
    "ParaMap.Obj.Emphysema",
    "ParaMap.Obj.fSAD",
    "ParaMap.Obj.Normal",
    "ParaMap.Obj.Uncategorized",
]

MASK_TYPES = APPLY_MASK_TYPES[1:]
# fmt: on

APS_TAG = [
    "apsLCS",
    "apsLungCOPD",
    "apsCAC",
    "apsILA",
    "apsLungTexture",
]

# 获取可执行程序所在实际绝对路径(项目打包时用)
BASE_DIR = path.dirname(path.realpath(argv[0]))
DEFAULT_LOGGER_CONFIG_FILE = BASE_DIR + "/config/simAPSMaskUpload_logging_config.ini"

# AVIEW sqlite db info
AVIEW_SQLITE_KEY = "corelinedb!@"
AVIEW_SQLITE_CIPHER = "aes-256-cbc"
AVIEW_SQLITE_KDF_ITER = 64000
AVIEW_SQLITE_CIPHER_PAGE_SIZE = 8192

# access-token 更新周期
CALLING_COUNT = 100
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJwYXlsb2FkIjp7ImVtYWlsIjoibWFza191cGxvYWRAc3VoMTIwLmNuIiwidXNlcm5hbWUiOiJzaW1NYXNrVXBsb2FkIiwiZGlzcGxheV9uYW1lIjoic2ltTWFza1VwbG9hZCIsInVpX2xhbmd1YWdlIjoiemgtQ04iLCJhY2Nlc3NfbGV2ZWwiOjUwMH0sImV4cCI6MzAwMDAwMDAwMCwiaWF0IjoxNTkwMTQ4MzkwfQ.8zO4iDyYymqCvz63geql3mFGyZu1O-WUBvdMHYblFNA"
