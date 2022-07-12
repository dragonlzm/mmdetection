# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .logger import get_root_logger
from .compat_config import compat_cfg

__all__ = ['get_root_logger', 'collect_env', 'compat_cfg']
