#!/usr/bin/env python3
"""验收：用真实社区 LoRA 验证往返无损与加载命中。

用法（文件自备，仓库不附带权重）：
    python acceptance.py \
        --akk   path/to/akk_4step_lora.safetensors \
        --animated path/to/animated_moss.safetensors \
        --comic path/to/american_comic.safetensors \
        --mflux-model path/to/mlx_vi_8bit \
        --web-dir path/to/web
    # 最后两项只有第 3 项「加载命中」需要：--web-dir 是 mflux024_adapter.py 所在目录

不给参数时只跑「格式自检」：随机张量造迷你 LoRA，往返后逐元素比较，
覆盖 qkv 拆分/合并、ffn 拆分/合并，以及拆不动时必须保留而不是丢弃。
没有 mlx 时自动回退 numpy 后端。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mosslora import convert, roundtrip_check, get_backend
from mosslora.convert import to_ir, normalize_ir, from_ir


def _to_numpy(t):
    """任意后端的张量 -> numpy float32，用于逐元素比较。"""
    import numpy as np
    try:
        import mlx.core as mx
        if isinstance(t, mx.array):
            return np.array(t.astype(mx.float32))
    except ImportError:
        pass
    return np.asarray(t, dtype=np.float32)


def _assert_equal_dict(got, want, what):
    """断言两个权重 dict 键集合相同、每个张量逐元素相等。"""
    import numpy as np
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    assert not missing and not extra, "%s 键集合不一致：缺 %s，多 %s" % (what, missing, extra)
    for k in want:
        a, b = _to_numpy(want[k]), _to_numpy(got[k])
        assert a.shape == b.shape, "%s 形状不一致：%s %s vs %s" % (what, k, a.shape, b.shape)
        assert np.array_equal(a, b), "%s 数值不一致：%s" % (what, k)


def _synthetic_check():
    """不需要真实权重：随机张量造迷你 LoRA，往返一遍并逐元素比较。

    必须用随机张量 —— 全零张量在 up/gate 切分顺序搞反时依然相等，等于没测。
    """
    import numpy as np

    be = get_backend()          # 有 mlx 用 mlx，没有自动回退 numpy
    rng = np.random.default_rng(0)
    try:
        import mlx.core as mx
        to_be = lambda a: mx.array(a)      # noqa: E731
    except ImportError:
        to_be = lambda a: a                # noqa: E731

    def mk(*shape):
        return to_be(rng.standard_normal(shape).astype(np.float32))

    # --- 1) mflux 布局：q/k/v/o + 融合的 ffn_up_gate + alpha，往返应逐元素相等
    w = {}
    for p in ("q", "k", "v", "o"):
        w["diffusion_model.layers.0.proj.%s.lora_down.weight" % p] = mk(8, 16)
        w["diffusion_model.layers.0.proj.%s.lora_up.weight" % p] = mk(16, 8)
    w["diffusion_model.layers.0.proj.ffn_up_gate.lora_down.weight"] = mk(8, 16)
    w["diffusion_model.layers.0.proj.ffn_up_gate.lora_up.weight"] = mk(32, 8)
    w["diffusion_model.layers.0.proj.ffn_down.lora_down.weight"] = mk(8, 32)
    w["diffusion_model.layers.0.proj.ffn_down.lora_up.weight"] = mk(16, 8)
    w["diffusion_model.layers.0.proj.ffn_up_gate.alpha"] = to_be(np.array(4.0, dtype=np.float32))

    ir, al, unk = to_ir(w, be)
    assert not unk, unk
    ir, warns, al = normalize_ir(ir, be, None, al)
    assert not warns, warns
    # ffn 32 行 -> up 16 + gate 16（up 在前）
    assert be.shape(ir[(0, "ffn_up", "up")])[0] == 16, "ffn up 拆分错误"
    assert be.shape(ir[(0, "ffn_gate", "up")])[0] == 16, "ffn gate 拆分错误"
    back = from_ir(ir, "mflux", be, al)
    _assert_equal_dict(back, w, "mflux 往返")

    # --- 2) qkv 融合（kohya 风格）：down 三份相同 + up 三倍行数 -> 拆成 q/k/v 再拼回去
    q = {}
    q["lora_unet_layers_0_proj_qkv.lora_down.weight"] = mk(8, 16)
    q["lora_unet_layers_0_proj_qkv.lora_up.weight"] = mk(48, 8)      # 3 * 16 行
    q["lora_unet_layers_0_proj_o.lora_down.weight"] = mk(8, 16)
    q["lora_unet_layers_0_proj_o.lora_up.weight"] = mk(16, 8)

    ir2, al2, unk2 = to_ir(q, be)
    assert not unk2, unk2
    ir2, warns2, al2 = normalize_ir(ir2, be, None, al2)
    assert not warns2, warns2
    full_up = _to_numpy(q["lora_unet_layers_0_proj_qkv.lora_up.weight"])
    full_down = _to_numpy(q["lora_unet_layers_0_proj_qkv.lora_down.weight"])
    for i, p in enumerate(("q", "k", "v")):
        assert np.array_equal(_to_numpy(ir2[(0, p, "up")]), full_up[i * 16:(i + 1) * 16]), \
            "qkv 拆分顺序错误：%s" % p
        assert np.array_equal(_to_numpy(ir2[(0, p, "down")]), full_down), "qkv down 未复制：%s" % p
    _assert_equal_dict(from_ir(ir2, "comfyui", be, al2), q, "qkv 往返")

    # --- 3) 边界：拆不动的融合张量必须原样保留，不能静默丢弃
    bad = {"lora_unet_layers_0_proj_qkv.lora_down.weight": mk(8, 16),
           "lora_unet_layers_0_proj_qkv.lora_up.weight": mk(8, 8)}    # 8 行，不能被 3 整除
    ir3, al3, _ = to_ir(bad, be)
    ir3, w3, _ = normalize_ir(ir3, be, None, al3)
    assert w3, "qkv 行数不可整除时应告警"
    assert (0, "qkv", "up") in ir3, "qkv 行数不可整除时不应丢弃张量"
    assert be.shape(ir3[(0, "qkv", "up")])[0] == 8, "保留的张量形状被改"

    # 半截 ffn：mflux 布局没有独立 proj 名可降级，必须报错而不是写成 (16,8) 的 ffn_up_gate
    half = {"diffusion_model.layers.0.proj.ffn_up.lora_down.weight": mk(8, 16),
            "diffusion_model.layers.0.proj.ffn_up.lora_up.weight": mk(16, 8)}
    ir4, al4, _ = to_ir(half, be)
    ir4, _w4, al4 = normalize_ir(ir4, be, None, al4)
    try:
        from_ir(ir4, "mflux", be, al4)
    except ValueError:
        pass
    else:
        raise AssertionError("半截 ffn 应报错，不能写出融合名 ffn_up_gate")

    return True, ("%d 键逐元素相等（ffn 32->16+16）；qkv 48->3x16 往返相等；"
                  "边界保留/报错 2 项" % len(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--akk", help="mflux 格式、含 alpha 与 ffn 融合的 LoRA")
    ap.add_argument("--animated", help="mflux 格式、无 alpha 的 LoRA")
    ap.add_argument("--comic", help="peft + 合并 qkv 的 LoRA")
    ap.add_argument("--mflux-model", help="mflux 模型目录（验证 apply_lora 命中数）")
    ap.add_argument("--web-dir", default=os.environ.get("MOSS_WEB_DIR"),
                    help="mflux024_adapter.py 所在目录（默认取环境变量 MOSS_WEB_DIR）")
    ap.add_argument("--mflux-site-packages", default=os.environ.get("MFLUX_SITE_PACKAGES"),
                    help="mflux 依赖所在 site-packages，会前置到子进程 sys.path"
                         "（默认取环境变量 MFLUX_SITE_PACKAGES；当前解释器已能 import mflux 时不用给）")
    a = ap.parse_args()

    be = get_backend()          # 有 mlx 用 mlx，没有回退 numpy（不再强制 mlx）
    ok_all = True

    def check(name, fn):
        nonlocal ok_all
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, "%s: %s" % (type(e).__name__, e)
        print("%-34s %s  %s" % (name, "PASS" if ok else "FAIL", msg))
        ok_all = ok_all and ok

    check("合成自检（随机张量往返）", _synthetic_check)

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
            web_dir = a.web_dir
            if not web_dir:
                return False, "缺 --web-dir（或环境变量 MOSS_WEB_DIR）"
            code = "import sys\nsys.path.insert(0, %r)\n" % web_dir
            if a.mflux_site_packages:
                code += "sys.path.insert(0, %r)\n" % a.mflux_site_packages
            code += ("import mflux024_adapter as A\n"
                     "dit = A.load_transformer(%r)\n"
                     "A.apply_lora(dit, ['/tmp/acc_comic.mflux.safetensors'], [1.0])\n"
                     % a.mflux_model)
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            out = p.stdout + p.stderr
            hit = [l for l in out.splitlines() if "keys matched" in l]
            if not hit:
                return False, "加载输出无命中数"
            m = re.search(r"(\d+)\s*/\s*(\d+)", hit[0])
            if not m:
                return False, "命中行无法解析: %s" % hit[0].strip()
            total = len(be.load("/tmp/acc_comic.mflux.safetensors"))
            got = int(m.group(1))
            if got != total:
                return False, "命中 %d/%s，但输出共 %d 键" % (got, m.group(2), total)
            return True, "%d/%d 命中" % (got, total)
        check("comic -> mflux 加载命中", t)

    print()
    print("总判定:", "全部通过" if ok_all else "有失败项")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
