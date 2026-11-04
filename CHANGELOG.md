# Changelog

## 0.1.1 — 2026-11-04

**10 条边界与自检缺陷。不改转换算法，只修边界处理与「自检其实测不出问题」。**

### Fixed — 自检

- `acceptance.py` 不再强制 mlx：`get_backend()` 自动回退 numpy，`pip install -e .`（不带 mlx extra）也能跑。
- 合成自检改成随机张量 + 逐元素断言，并补了 qkv 融合样例。原来用全零张量只断言形状，把 up/gate 顺序写反也照样「通过」。
- acceptance 的加载命中路径走 `--web-dir` / `MOSS_WEB_DIR`，并断言命中数等于总键数。

### Fixed — 转换边界

- mflux 目标遇半截 ffn 直接报错，不再把 `(16,8)` 写成 `ffn_up_gate`、也不再后写覆盖丢张量。
- qkv 行数不能被 3 整除时原样保留融合张量并告警，不再整块丢弃（8 行 qkv 曾导致输出 0 键）。
- `ffn_gate_up` 按名字顺序切分（gate 在前），不再一律 up 在前。
- `emit_key` 对未知 proj 用 `.get(proj, proj)`，`proj.attn` 不再 KeyError。
- `_default_dst` 与源文件同名时拒绝（同名同格式转换会原地覆盖源文件）。
- CLI 有未识别键时退出码 1；删掉 docstring 里不存在的 `--self-test`。
- ComfyUI Inspect 补占位符守卫；投影名复用 `to_ir` 解析（kohya 下划线键曾恒为空）。

### Changed

- `normalize_ir` 保留未拆开的融合件 alpha；`convert` 的 warnings 统计覆盖融合阶段。

## 0.1.0 — 2026-11-04

首个公开版本：ComfyUI / kohya / peft ↔ mflux 双向转换，无损（往返逐元素相等），不碰权重数值。
