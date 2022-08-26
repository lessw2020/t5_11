
from . import default_hooks as default

from .default_hooks import fp16_compress_hook, bf16_compress_hook

LOW_PRECISION_HOOKS = [
    default.fp16_compress_hook,
    default.bf16_compress_hook,
]
