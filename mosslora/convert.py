"""MOSS-Image LoRA 双向转换：ComfyUI/kohya/peft  <->  mflux。

核心是"拆到最细粒度再拼回去"：
  qkv 融合的 up 张量按 3 等分拆成 q/k/v（down 复制三份）
  ffn_up_gate 融合的 up 张量按 2 等分拆成 up/gate（up 在前，与 mflux 的 mx.split(...,2) 一致）
反向则把 q/k/v 拼回 qkv、把 up/gate 拼回 ffn_up_gate。

张量后端：优先 mlx（MOSS 生态必备，能读写 bf16），否则回退 safetensors.numpy。
"""
from __future__ import annotations

import os
from collections import OrderedDict

from .formats import detect, parse_key, emit_key, CANON_PROJ, MFLUX_PROJ


# ------------------------------------------------------------------ 后端

class Backend:
    name = "?"

    def load(self, path):
        raise NotImplementedError

    def save(self, path, tensors):
        raise NotImplementedError

    def concat(self, xs, axis=0):
        raise NotImplementedError

    def split(self, x, n, axis=0):
        raise NotImplementedError

    def shape(self, x):
        raise NotImplementedError

    def scalar(self, v):
        """标量（如 LoRA 的 alpha）转成本后端的 0 维数组。"""
        raise NotImplementedError


class MLXBackend(Backend):
    name = "mlx"

    def load(self, path):
        import mlx.core as mx
        return dict(mx.load(path))

    def save(self, path, tensors):
        import mlx.core as mx
        mx.save_safetensors(path, {k: v for k, v in tensors.items()})

    def concat(self, xs, axis=0):
        import mlx.core as mx
        return mx.concatenate(xs, axis=axis)

    def split(self, x, n, axis=0):
        import mlx.core as mx
        return mx.split(x, n, axis=axis)

    def shape(self, x):
        return tuple(x.shape)

    def scalar(self, v):
        import mlx.core as mx
        return mx.array(float(v), dtype=mx.float32)


class NumpyBackend(Backend):
    name = "numpy"

    def load(self, path):
        from safetensors.numpy import load_file
        return load_file(path)

    def save(self, path, tensors):
        from safetensors.numpy import save_file
        save_file({k: v for k, v in tensors.items()}, path)

    def concat(self, xs, axis=0):
        import numpy as np
        return np.concatenate(xs, axis=axis)

    def split(self, x, n, axis=0):
        import numpy as np
        return np.split(x, n, axis=axis)

    def shape(self, x):
        return tuple(x.shape)

    def scalar(self, v):
        import numpy as np
        return np.array(float(v), dtype=np.float32)


def get_backend(prefer=None):
    if prefer in (None, "mlx"):
        try:
            import mlx.core  # noqa: F401
            return MLXBackend()
        except ImportError:
            if prefer == "mlx":
                raise
    return NumpyBackend()


# ------------------------------------------------------------------ 解析到 IR

def to_ir(weights, be):
    """权重 dict -> {(layer, canon_proj, 'down'|'up'): tensor}。

    同时收集 alpha（标量缩放），单独返回。
    """
    ir = OrderedDict()
    alphas = {}
    unknown = []
    for key, val in weights.items():
        parsed = parse_key(key)
        if parsed is None:
            if key.endswith(".alpha"):
                # 从 alpha 键反推 layer/proj
                base = key[:-len(".alpha")]
                p2 = parse_key(base + ".lora_down.weight")
                if p2:
                    alphas[(p2[0], p2[1])] = float(val)
                continue
            unknown.append(key)
            continue
        layer, proj, kind = parsed
        ir[(layer, proj, kind)] = val
    return ir, alphas, unknown


def normalize_ir(ir, be, log=None, alphas=None):
    """把融合布局（qkv / ffn_up_gate）拆成最细粒度。幂等。

    alphas 是 {(layer, proj): float}，跟 proj 一起拆（融合件的 alpha 复制给拆出来的每个 proj）。
    """
    out = OrderedDict()
    warns = []
    alphas_out = dict(alphas or {})

    # 先收集每层的融合项
    fused = {}
    for (layer, proj, kind), t in ir.items():
        if proj in ("qkv", "ffn_up_gate", "ffn_gate_up", "up_gate"):
            fused.setdefault((layer, proj), {})[kind] = t
        else:
            out[(layer, proj, kind)] = t

    for (layer, proj), parts in fused.items():
        up = parts.get("up")
        down = parts.get("down")
        if up is None:
            warns.append("层 %d 的 %s 缺 lora_up，跳过" % (layer, proj))
            continue
        rows = be.shape(up)[0]
        if proj == "qkv":
            if rows % 3:
                warns.append("层 %d 的 qkv up 行数 %d 不能被 3 整除，跳过" % (layer, rows))
                continue
            q, k, v = be.split(up, 3, axis=0)
            for name, t in (("q", q), ("k", k), ("v", v)):
                out[(layer, name, "up")] = t
                if down is not None:
                    out[(layer, name, "down")] = down
                if (layer, "qkv") in alphas_out:
                    alphas_out[(layer, name)] = alphas_out[(layer, "qkv")]
        else:  # ffn_up_gate：mflux 里 mx.split(x, 2) -> 前 up、后 gate
            if rows % 2:
                warns.append("层 %d 的 %s up 行数 %d 不能被 2 整除，跳过" % (layer, proj, rows))
                continue
            up_half, gate_half = be.split(up, 2, axis=0)
            out[(layer, "ffn_up", "up")] = up_half
            out[(layer, "ffn_gate", "up")] = gate_half
            if down is not None:
                out[(layer, "ffn_up", "down")] = down
                out[(layer, "ffn_gate", "down")] = down
            if (layer, proj) in alphas_out:
                alphas_out[(layer, "ffn_up")] = alphas_out[(layer, proj)]
                alphas_out[(layer, "ffn_gate")] = alphas_out[(layer, proj)]

    if log:
        for w in warns:
            log(w)
    # 融合名不再保留
    for k in list(alphas_out):
        if k[1] in ("qkv", "ffn_up_gate", "ffn_gate_up", "up_gate"):
            alphas_out.pop(k)
    return out, warns, alphas_out


def fuse_for_mflux(ir, be, log=None):
    """IR -> mflux 布局：ffn_up + ffn_gate 合并成 ffn_up_gate（up 在前、gate 在后）。"""
    out = OrderedDict()
    per_layer = {}
    for (layer, proj, kind), t in ir.items():
        per_layer.setdefault(layer, {})[(proj, kind)] = t

    for layer, d in sorted(per_layer.items()):
        # ffn：需要 up 和 gate 都在才能融合
        has_up = ("ffn_up", "up") in d
        has_gate = ("ffn_gate", "up") in d
        if has_up and has_gate:
            up_t = be.concat([d[("ffn_up", "up")], d[("ffn_gate", "up")]], axis=0)
            down_t = d.get(("ffn_up", "down"))
            gate_down = d.get(("ffn_gate", "down"))
            if down_t is not None and gate_down is not None:
                a, b = be.shape(down_t), be.shape(gate_down)
                if a != b and log:
                    log("层 %d: ffn_up/ffn_gate 的 lora_down 形状不同 %s vs %s，取 ffn_up 的" % (layer, a, b))
            out[(layer, "ffn_up_gate", "up")] = up_t
            if down_t is not None:
                out[(layer, "ffn_up_gate", "down")] = down_t
            # 其余 proj（q/k/v/o/ffn_down）原样保留——之前这里漏了，
            # 导致 mflux 输出只剩 ffn_up_gate 的 2 个键/层（42 层 = 84 键）。
            for (proj, kind), t in d.items():
                if proj not in ("ffn_up", "ffn_gate"):
                    out[(layer, proj, kind)] = t
        else:
            for (proj, kind), t in d.items():
                if proj in ("ffn_up", "ffn_gate"):
                    if log:
                        log("层 %d: 只有 %s 没有配对的一半，保留为独立 proj（mflux 可能忽略）" % (layer, proj))
                    out[(layer, proj, kind)] = t
                else:
                    out[(layer, proj, kind)] = t
    return out


def _same_tensors(ts, be):
    import numpy as np

    def _np(t):
        try:
            import mlx.core as mx
            if isinstance(t, mx.array):
                return np.array(t.astype(mx.float32))
        except ImportError:
            pass
        return np.asarray(t, dtype=np.float32)

    first = _np(ts[0])
    return all(np.array_equal(first, _np(t)) for t in ts[1:])


def fuse_qkv(ir, be, log=None):
    """IR -> q/k/v 合并成 qkv（ComfyUI 风格）。down 侧不同时取 q 的并告警。"""
    per_layer = {}
    for (layer, proj, kind), t in ir.items():
        per_layer.setdefault(layer, {})[(proj, kind)] = t

    out = OrderedDict()
    fused_layers = 0
    for layer, d in sorted(per_layer.items()):
        downs = [d.get((p, "down")) for p in ("q", "k", "v")]
        can_fuse = (all((p, "up") in d for p in ("q", "k", "v"))
                    and all(x is not None for x in downs)
                    and _same_tensors(downs, be))
        if can_fuse:
            ups = [d[(p, "up")] for p in ("q", "k", "v")]
            out[(layer, "qkv", "up")] = be.concat(ups, axis=0)
            out[(layer, "qkv", "down")] = downs[0]
            fused_layers += 1
            for p in ("q", "k", "v"):
                d.pop((p, "up"), None)
                d.pop((p, "down"), None)
        elif all((p, "up") in d for p in ("q", "k", "v")) and log:
            # q/k/v 的 down 各不相同 -> 融合会丢信息，保持分开（ComfyUI 也认分开的 proj_q/k/v）
            log("层 %d: q/k/v 的 lora_down 不同，不融合 qkv（保持分开，无损）" % layer)
        for (proj, kind), t in d.items():
            out[(layer, proj, kind)] = t
    if log and fused_layers:
        log("已融合 %d 层的 qkv" % fused_layers)
    return out


def from_ir(ir, style, be, alphas=None):
    """IR -> 目标风格的权重 dict。

    style: 'mflux'（q/k/v 分开 + ffn 融合）
           'comfyui'（qkv 融合 + ffn 分开，键名走 kohya 下划线）
           'kohya' / 'peft'（全部最细粒度）
    """
    if style == "mflux":
        ir = fuse_for_mflux(ir, be)
    elif style == "comfyui":
        ir = fuse_qkv(ir, be)
        style = "kohya"
    out = OrderedDict()
    for (layer, proj, kind), t in sorted(ir.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        out[emit_key(style, layer, proj, kind)] = t
    if alphas:
        seen = set()
        for (layer, proj), a in alphas.items():
            if proj not in CANON_PROJ:
                continue
            k = emit_key(style, layer, proj, "alpha")
            if k in seen:      # ffn_up / ffn_gate 在 mflux 里映射到同一个 ffn_up_gate
                continue
            seen.add(k)
            out[k] = be.scalar(a)
    return out


# ------------------------------------------------------------------ 顶层 API

def convert(src, dst=None, to="mflux", backend=None, verbose=True):
    """把一个 LoRA 文件转成目标格式。返回统计 dict。"""
    be = get_backend(backend)
    log = (lambda s: print("[warn] " + s)) if verbose else None

    weights = be.load(src)
    info = detect(weights.keys())
    ir, alphas, unknown = to_ir(weights, be)
    ir, warns, alphas = normalize_ir(ir, be, log, alphas)

    dst = dst or _default_dst(src, to)
    out = from_ir(ir, to, be, alphas)
    be.save(dst, out)

    stats = {
        "src": src, "dst": dst, "backend": be.name,
        "src_format": info["style"], "target": to,
        "src_keys": len(weights), "dst_keys": len(out),
        "layers": len({k[0] for k in ir}), "projs": len({k[1] for k in ir}),
        "unknown_keys": len(unknown), "warnings": len(warns),
    }
    if verbose:
        print("源格式 %-8s (%d 键)  ->  %-8s (%d 键)  层 %d / proj %d"
              % (stats["src_format"], stats["src_keys"], to, stats["dst_keys"],
                 stats["layers"], stats["projs"]))
        if unknown:
            print("[warn] %d 个键无法识别，例如 %s" % (len(unknown), unknown[:3]))
        print("写出:", dst)
    return stats


def _default_dst(src, to):
    base, ext = os.path.splitext(src)
    for tag in (".mflux", ".kohya", ".peft"):
        if base.endswith(tag):
            base = base[:-len(tag)]
    return base + "." + to + ext


def roundtrip_check(a, b, be=None):
    """比较两个文件的所有张量是否逐元素相等（用于往返验证）。"""
    be = be or get_backend()
    x, y = be.load(a), be.load(b)
    if set(x) != set(y):
        return False, "键集合不同: 只在 A 有 %d 个, 只在 B 有 %d 个" % (
            len(set(x) - set(y)), len(set(y) - set(x)))
    import numpy as np

    def _np(t):
        try:
            import mlx.core as mx
            if isinstance(t, mx.array):
                return np.array(t.astype(mx.float32))
        except ImportError:
            pass
        return np.asarray(t, dtype=np.float32)

    for k in x:
        xa, ya = _np(x[k]), _np(y[k])
        if xa.shape != ya.shape or not np.array_equal(xa, ya):
            return False, "张量不一致: %s %s vs %s" % (k, xa.shape, ya.shape)
    return True, "全部 %d 个张量逐元素相等" % len(x)
