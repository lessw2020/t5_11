
#from . import default_hooks as default

from .default_hooks import fp16_compress_hook, bf16_compress_hook, default_hooks

LOW_PRECISION_HOOKS = [
    fp16_compress_hook,
    bf16_compress_hook,
]
