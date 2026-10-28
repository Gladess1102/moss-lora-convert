#!/usr/bin/env python3
"""验收：用真实社区 LoRA 验证往返无损与加载命中。

用法（文件自备，仓库不附带权重）：
    python acceptance.py \
        --akk   path/to/akk_4step_lora.safetensors \
        --animated path/to/animated_moss.safetensors \
        --comic path/to/american_comic.safetensors \
        --mflux-model path/to/mlx_vi_8bit          # 仅第 3 项加载验证需要

不给参数时只跑「格式自检」：用内置合成张量验证 qkv 拆分/合并与 ffn 拆分/合并。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mosslora import convert, roundtrip_check, get_backend
from mosslora.convert import to_ir, normalize_ir, from_ir


def _synthetic_check():
    """不需要真实权重：构造一个 mflux 布局的迷你 LoRA，往返一遍。"""
    be = get_backend("mlx")
    try:
        import mlx.core as mx
        mk = lambda *s: mx.zeros(s, dtype=mx.float32)  # noqa: E731
    except ImportError:
        import numpy as np
        mk = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
    # 1 层：q/k/v/o 各 down[8,16] up[16,8]，ffn_up_gate up[32,8] down[8,16]，ffn_down 反向
    w = {}
    for p in ("q", "k", "v", "o"):
        w["diffusion_model.layers.0.proj.%s.lora_down.weight" % p] = mk(8, 16)
        w["diffusion_model.layers.0.proj.%s.lora_up.weight" % p] = mk(16, 8)
    w["diffusion_model.layers.0.proj.ffn_up_gate.lora_down.weight"] = mk(8, 16)
    w["diffusion_model.layers.0.proj.ffn_up_gate.lora_up.weight"] = mk(32, 8)
    w["diffusion_model.layers.0.proj.ffn_down.lora_down.weight"] = mk(8, 32)
    w["diffusion_model.layers.0.proj.ffn_down.lora_up.weight"] = mk(16, 8)

    ir, al, unk = to_ir(w, be)
    ir, warns, al = normalize_ir(ir, be, None, al)
    assert not unk, unk
    # ffn 拆分：32 行 -> 16 + 16
    assert be.shape(ir[(0, "ffn_up", "up")])[0] == 16, "ffn up 拆分错误"
    assert be.shape(ir[(0, "ffn_gate", "up")])[0] == 16, "ffn gate 拆分错误"
    back = from_ir(ir, "mflux", be, al)
    assert len(back) == len(w), "键数不一致 %d vs %d" % (len(back), len(w))
    print("合成自检                              PASS  ffn 32->16+16，往返键数一致")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--akk", help="mflux 格式、含 alpha 与 ffn 融合的 LoRA")
    ap.add_argument("--animated", help="mflux 格式、无 alpha 的 LoRA")
    ap.add_argument("--comic", help="peft + 合并 qkv 的 LoRA")
    ap.add_argument("--mflux-model", help="mflux 模型目录（验证 apply_lora 命中数）")
    a = ap.parse_args()

    be = get_backend("mlx")
    ok_all = True

    def check(name, fn):
        nonlocal ok_all
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, "%s: %s" % (type(e).__name__, e)
        print("%-34s %s  %s" % (name, "PASS" if ok else "FAIL", msg))
        ok_all = ok_all and ok

    _synthetic_check()

    if a.akk:
        def t():
            convert(a.akk, "/tmp/acc_akk.comfy.safetensors", to="comfyui")
            convert("/tmp/acc_akk.comfy.safetensors", "/tmp/acc_akk.back.safetensors", to="mflux")
            return roundtrip_check(a.akk, "/tmp/acc_akk.back.safetensors", be)
        check("akk: mflux->comfyui->mflux", t)

    if a.animated:
        def t():
            convert(a.animated, "/tmp/acc_ani.kohya.safetensors", to="kohya")
            convert("/tmp/acc_ani.kohya.safetensors", "/tmp/acc_ani.back.safetensors", to="mflux")
            return roundtrip_check(a.animated, "/tmp/acc_ani.back.safetensors", be)
        check("animated: mflux->kohya->mflux", t)

    if a.comic:
        def t():
            st = convert(a.comic, "/tmp/acc_comic.mflux.safetensors", to="mflux")
            return st["dst_keys"] > 0, "输出 %d 键" % st["dst_keys"]
        check("comic: peft+qkv -> mflux", t)

    if a.comic and a.mflux_model:
        def t():
            import subprocess
            code = (
                "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
                "import mflux024_adapter as A\n"
                "dit = A.load_transformer(%r)\n"
                "A.apply_lora(dit, ['/tmp/acc_comic.mflux.safetensors'], [1.0])\n"
                % (os.path.join(os.path.dirname(a.mflux_model), "..", "web"),
                   os.path.join(os.path.dirname(a.mflux_model), "..", "venv-mflux024",
                                "lib", "python3.13", "site-packages"),
                   a.mflux_model)
            )
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            out = p.stdout + p.stderr
            hit = [l for l in out.splitlines() if "keys matched" in l]
            if not hit:
                return False, "加载输出无命中数"
            return True, hit[0].strip().split("(")[-1].split(" ")[0]
        check("comic -> mflux 加载命中", t)

    print()
    print("总判定:", "全部通过" if ok_all else "有失败项")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
