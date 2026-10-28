"""moss-lora-convert 命令行。

  moss-lora-convert in.safetensors --to mflux
  moss-lora-convert in.safetensors --to comfyui -o out.safetensors
  moss-lora-convert in.safetensors --inspect
  moss-lora-convert --self-test
"""
from __future__ import annotations

import argparse
import sys

from .convert import convert, get_backend, roundtrip_check
from .formats import detect


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="moss-lora-convert",
        description="MOSS-Image LoRA 格式互转：ComfyUI/kohya/peft <-> mflux")
    ap.add_argument("src", help="输入 .safetensors")
    ap.add_argument("-o", "--out", help="输出路径（默认 <src>.<target>.safetensors）")
    ap.add_argument("--to", default="mflux",
                    choices=["mflux", "comfyui", "kohya", "peft"],
                    help="目标格式（默认 mflux）")
    ap.add_argument("--backend", default=None, choices=["mlx", "numpy"],
                    help="张量后端；默认自动（有 mlx 用 mlx，能处理 bf16）")
    ap.add_argument("--inspect", action="store_true", help="只检测格式，不转换")
    ap.add_argument("--quiet", "-q", action="store_true")
    a = ap.parse_args(argv)

    be = get_backend(a.backend)
    if a.inspect:
        w = be.load(a.src)
        info = detect(w.keys())
        print("文件: %s" % a.src)
        print("键数: %d" % len(w))
        print("风格: %s（LoRA 命名 %s，前缀 %r）"
              % (info["style"], info["lora_naming"], info["prefix"]))
        from .convert import to_ir, normalize_ir
        ir, alphas, unknown = to_ir(w, be)
        ir, warns, alphas = normalize_ir(ir, be, None, alphas)
        print("层数: %d" % len({k[0] for k in ir}))
        print("投影: %s" % sorted({k[1] for k in ir}))
        print("alpha: %d 个" % len(alphas))
        if unknown:
            print("未识别键: %d（例如 %s）" % (len(unknown), unknown[:3]))
        return 0

    stats = convert(a.src, a.out, to=a.to, backend=a.backend, verbose=not a.quiet)
    return 0 if stats["unknown_keys"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
