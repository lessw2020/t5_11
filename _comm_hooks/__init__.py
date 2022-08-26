
from . import default_hooks as def_hooks

from .default_hooks import *  # fp16_compress_hook, bf16_compress_hook, default_hooks

LOW_PRECISION_HOOKS = [
    def_hooks.fp16_compress_hook,
    def_hooks.bf16_compress_hook,
]
