import logging
import os
import warnings

from base.config import single_config as config

# =============================================
# 抑制第三方库的噪音日志和警告
# =============================================
_THIRD_PARTY_LOGGERS = {
    'transformers': logging.WARNING,
    'transformers.tokenization_utils': logging.ERROR,
    'transformers.tokenization_utils_base': logging.ERROR,
    'sentence_transformers': logging.WARNING,
    'jieba': logging.WARNING,
    'jieba.finalseg': logging.ERROR,
    'pymilvus': logging.WARNING,
    'urllib3': logging.WARNING,
    'urllib3.connectionpool': logging.WARNING,
    'filelock': logging.WARNING,
    'datasets': logging.WARNING,
    'huggingface_hub': logging.WARNING,
}

for _name, _level in _THIRD_PARTY_LOGGERS.items():
    logging.getLogger(_name).setLevel(_level)

# 抑制 transformers 等库的 FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, module=r'transformers.*')
warnings.filterwarnings('ignore', category=UserWarning, module=r'transformers.*')


# =============================================
# 日志格式与处理器配置
# =============================================

# 优化后的日志格式：用 | 分隔，%(name)s 显示层级名称如 EDU_RAG.mysql_client
_LOG_FORMAT = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def _create_handlers(log_url, console_level='INFO', encode='utf-8', mode='a'):
    """创建控制台和文件处理器"""
    os.makedirs(os.path.dirname(log_url), exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level, logging.INFO))
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_url, mode=mode, encoding=encode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    return console_handler, file_handler


def _init_root_logger():
    """初始化根日志记录器 EDU_RAG，仅执行一次"""
    root_logger = logging.getLogger('EDU_RAG')
    root_logger.setLevel(logging.DEBUG)

    if not root_logger.handlers:
        console_handler, file_handler = _create_handlers(config.LOG_FILE, config.LOG_CONSOLE_LEVEL)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    return root_logger


# 初始化根 logger
_root_logger = _init_root_logger()


def get_module_logger(module_name):
    """
    获取模块级别的命名日志记录器

    用法：
        from base.logger import get_module_logger
        logger = get_module_logger(__name__)

    生成的日志名称示例：
        - __name__ = 'mysql_qa.db.mysql_client' → logger名 = 'EDU_RAG.mysql_client'
        - __name__ = 'rag_qa.core.rag_system'  → logger名 = 'EDU_RAG.rag_system'
        - __name__ = 'old_main'                 → logger名 = 'EDU_RAG.old_main'

    :param module_name: 通常传入 __name__
    :return: logging.Logger 实例
    """
    # 从完整的模块路径中提取最后一级作为子 logger 名称
    # 例如 'mysql_qa.db.mysql_client' → 'mysql_client'
    short_name = module_name.rsplit('.', 1)[-1] if '.' in module_name else module_name
    return logging.getLogger(f'EDU_RAG.{short_name}')


# =============================================
# 向后兼容：保留原有的 single_logger
# =============================================
single_logger = get_module_logger('main')
