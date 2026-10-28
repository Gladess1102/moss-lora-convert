"""MOSS-Image LoRA 的键名格式：检测、解析、生成。

社区里同一个 LoRA 会有三种写法，张量布局也不一样：

    mflux / DiffSynth   diffusion_model.layers.0.proj.q.lora_down.weight
                        proj.q / proj.k / proj.v / proj.o / proj.ffn_up_gate / proj.ffn_down
                        q/k/v 分开，ffn 的 up+gate 融合（输出 2*ffn_dim，up 在前 gate 在后）

    kohya (ComfyUI)     lora_unet_layers_0_proj_q.lora_down.weight
                        proj_q / proj_k / proj_v / proj_o / proj_ffn_gate / proj_ffn_up / proj_ffn_down
                        ffn 分开，部分版本把 q/k/v 融合成 qkv

    peft (ai-toolkit)   transformer.layers.0.proj.q.lora_A.weight
                        前缀 transformer. / diffusion_model.，命名 lora_A/lora_B

统一成最细粒度 IR（q/k/v 分开、ffn_up/ffn_gate 分开），再按目标格式拼回去。
"""
from __future__ import annotations

import re

# 最细粒度的投影名
CANON_PROJ = ("q", "k", "v", "o", "ffn_up", "ffn_gate", "ffn_down")

# mflux 侧实际的 proj 键（ffn_up + ffn_gate 融合成 ffn_up_gate）
MFLUX_PROJ = {"q": "q", "k": "k", "v": "v", "o": "o",
              "ffn_up": "ffn_up_gate", "ffn_gate": "ffn_up_gate", "ffn_down": "ffn_down",
              # 已经是融合名的自映射，避免 emit_key 对未知 proj 抛 KeyError
              "ffn_up_gate": "ffn_up_gate", "qkv": "qkv"}

_RE_LAYER_DOT = re.compile(r"(?:^|\.)(?:transformer|diffusion_model)\.layers\.(\d+)\.proj\.([a-zA-Z_]+)\.(lora_down|lora_up|lora_A|lora_B)(?:\.weight)?$")
_RE_LAYER_DOT_NOPREFIX = re.compile(r"^layers\.(\d+)\.proj\.([a-zA-Z_]+)\.(lora_down|lora_up|lora_A|lora_B)(?:\.weight)?$")
_RE_LAYER_UNDER = re.compile(r"(?:^|_)lora_unet_layers_(\d+)_proj_([a-zA-Z_]+)\.(lora_down|lora_up)(?:\.weight)?$")


def detect(keys):
    """识别 LoRA 的键名风格。返回 dict（不猜张量布局，布局在解析时按形状决定）。"""
    ks = list(keys)
    if not ks:
        return {"style": "unknown", "lora_naming": None, "prefix": None}
    n_kohya = sum(1 for k in ks if "lora_unet_" in k)
    n_peft = sum(1 for k in ks if re.search(r"\.lora_[AB]\.", k))
    n_mflux = sum(1 for k in ks if k.startswith("diffusion_model.") and ".lora_" in k)
    if n_kohya:
        return {"style": "kohya", "lora_naming": "lora_down/up", "prefix": "lora_unet_"}
    if n_peft:
        return {"style": "peft", "lora_naming": "lora_A/B",
                "prefix": "transformer." if any(k.startswith("transformer.") for k in ks) else
                          ("diffusion_model." if n_mflux else "")}
    if n_mflux:
        return {"style": "mflux", "lora_naming": "lora_down/up", "prefix": "diffusion_model."}
    return {"style": "unknown", "lora_naming": None, "prefix": None}


def parse_key(key):
    """把任意风格的键解析成 (layer, proj, 'down'|'up')，失败返回 None。"""
    for rx in (_RE_LAYER_DOT, _RE_LAYER_DOT_NOPREFIX):
        m = rx.search(key)
        if m:
            layer, proj, kind = int(m.group(1)), m.group(2), m.group(3)
            return layer, proj, ("down" if kind in ("lora_down", "lora_A") else "up")
    m = _RE_LAYER_UNDER.search(key)
    if m:
        layer, proj, kind = int(m.group(1)), m.group(2), m.group(3)
        return layer, proj, ("down" if kind == "lora_down" else "up")
    return None


def emit_key(style, layer, proj, kind, with_weight=True):
    """按目标风格生成键名。proj 用 IR 的规范名（q/k/v/o/ffn_up/ffn_gate/ffn_down）。"""
    suffix = ".weight" if with_weight else ""
    if kind == "alpha":
        if style == "kohya":
            return "lora_unet_layers_%d_proj_%s.alpha" % (layer, proj)
        if style == "peft":
            return "transformer.layers.%d.proj.%s.alpha" % (layer, proj)
        return "diffusion_model.layers.%d.proj.%s.alpha" % (layer, MFLUX_PROJ[proj])
    if style == "kohya":
        k = "lora_down" if kind == "down" else "lora_up"
        return "lora_unet_layers_%d_proj_%s.%s%s" % (layer, proj, k, suffix)
    if style == "peft":
        k = "lora_A" if kind == "down" else "lora_B"
        return "transformer.layers.%d.proj.%s.%s%s" % (layer, proj, k, suffix)
    # mflux
    k = "lora_down" if kind == "down" else "lora_up"
    return "diffusion_model.layers.%d.proj.%s.%s%s" % (layer, MFLUX_PROJ[proj], k, suffix)
