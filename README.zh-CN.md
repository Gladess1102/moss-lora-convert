# moss-lora-convert

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Backend](https://img.shields.io/badge/backend-MLX%20%7C%20numpy-lightgrey.svg)](#安装)

[English](README.md) · **简体中文**

**MOSS-Image LoRA 格式互转**：ComfyUI / kohya / peft ↔ mflux。

社区里下载的 MOSS LoRA，十有八九直接喂给 mflux 会报 `No LoRA layers were applied`。不是文件坏了，是键名和张量布局不一样。这个工具做两件事：

1. **改名** —— `lora_unet_layers_0_proj_q` / `transformer.layers.0.proj.q` → `diffusion_model.layers.0.proj.q`
2. **重排张量** —— 把融合的 QKV 拆成 q/k/v、把融合的 ffn_up_gate 拆成 up/gate（以及反向合并）

转换是**无损**的（往返逐元素相等，见[自测](#自测)），且**不碰权重数值**，只改键名和切片。

![转换前后](docs/before-after.png)

*同 prompt、同 seed、同 LoRA、同 scale 0.8。**左：** 原始的 ComfyUI 风格文件，mflux 只命中 246/328 个键，QKV 层被静默丢弃。**右：** 跑完 `moss-lora-convert --to mflux` 后 492/492 全命中，注意力层才真正参与风格。*

---

## 为什么需要它

MOSS-Image 的 DiT 在每个 block 里有 6 个投影：`q, k, v, o, ffn_up_gate, ffn_down`。但不同生态的 LoRA 把它们的**存储布局**排得不一样：

| | q / k / v | FFN | 键名风格 |
|---|---|---|---|
| **mflux** / DiffSynth | 分开 | `ffn_up_gate` 融合（输出 2×ffn_dim，up 在前 gate 在后） | `diffusion_model.layers.N.proj.X.lora_down/up.weight` |
| **kohya**（ComfyUI 常见） | 常融合成 `qkv` | `ffn_gate` + `ffn_up` 分开 | `lora_unet_layers_N_proj_X.lora_down/up.weight` |
| **peft**（ai-toolkit） | 分开或融合 | 视训练脚本 | `transformer.layers.N.proj.X.lora_A/B.weight` |

所以光改名不够——**融合的张量必须按正确比例切开**：

```
qkv.lora_up         [9216, r]  ->  q [3072] + k [3072] + v [3072]
ffn_up_gate.lora_up [16384, r] ->  up [8192] + gate [8192]   # up 在前
```

切错顺序或比例，加载不会报错，但**风格会静默失效或错乱**。

---

## 安装

```bash
pip install -e .            # 只要 CLI 和库
pip install -e ".[mlx]"     # 有 MLX（MOSS 生态）时推荐：能直接读写 bf16
```

不需要 torch。张量后端优先 MLX，没有就回退 `safetensors.numpy`（fp16/fp32 够用，bf16 会报错）。

---

## 用法

### 命令行

```bash
# 看它到底是什么格式（不写文件）
moss-lora-convert my_lora.safetensors --inspect

# 转成 mflux 能直接加载的
moss-lora-convert my_lora.safetensors --to mflux

# 转回 ComfyUI / kohya
moss-lora-convert my_lora.mflux.safetensors --to comfyui -o my_lora.comfy.safetensors

# 指定输出
moss-lora-convert in.safetensors --to peft -o out.safetensors
```

`--inspect` 的输出长这样：

```
文件: american_comic.safetensors
键数: 328
风格: peft（LoRA 命名 lora_A/B，前缀 'diffusion_model.'）
层数: 41
投影: ['ffn_down', 'ffn_gate', 'ffn_up', 'k', 'o', 'q', 'v']
alpha: 0 个
```

### ComfyUI 节点

把整个目录放进 `ComfyUI/custom_nodes/`（或做软链接），重启后得到两个节点，分类在 **MOSS/LoRA**：

- **MOSS LoRA Convert** —— 选 `models/loras` 里的文件 + 目标格式，输出转换后的绝对路径
- **MOSS LoRA Inspect** —— 只检测格式，返回一份文本报告

### Python

```python
from mosslora import convert

stats = convert("my_lora.safetensors", "my_lora.mflux.safetensors", to="mflux")
print(stats)   # {'src_format': 'kohya', 'src_keys': 756, 'dst_keys': 756, 'layers': 42, ...}
```

---

## 支持矩阵

| 输入 → 输出 | mflux | comfyui | kohya | peft |
|---|---|---|---|---|
| **mflux** | — | ✅ 无损 | ✅ 无损 | ✅ 无损 |
| **kohya** | ✅ 无损 | ✅ 无损 | — | ✅ 无损 |
| **peft** | ✅ 无损 | ✅ 无损 | ✅ 无损 | — |
| **融合 qkv** | ✅ 拆分（无损） | ✅ 保持融合 | ✅ 拆开 | ✅ 拆开 |
| **融合 ffn_up_gate** | ✅ 保持 | ✅ 拆开 | ✅ 拆开 | ✅ 拆开 |

「无损」指往返转换后**逐元素相等**，见下方自测。

**关于 qkv 融合的一个细节**：从 mflux 转 comfyui 时，只有当某层 q/k/v 的 `lora_down` 完全相同时才会融合。不同就保持分开（ComfyUI 也认分开的 `proj_q`/`proj_k`/`proj_v`），并打一行警告——因为融合会丢掉另外两份 down，是有损的。

---

## 自测

```bash
# 合成自检 —— 不需要任何权重
python acceptance.py

# 用真实社区 LoRA 跑完整验收
python acceptance.py \
    --akk path/to/akk_4step_lora.safetensors \
    --animated path/to/animated_moss.safetensors \
    --comic path/to/american_comic.safetensors
```

实测（2026-10，三份真实社区 LoRA）：

| 文件 | 原始 | 往返 | 结果 |
|---|---|---|---|
| akhaliq 4 步蒸馏 | mflux，756 键（含 alpha，ffn 融合） | mflux→kohya→mflux | **756 张量逐元素相等** |
| axiomofmind 动画风 | mflux，504 键（无 alpha） | mflux→kohya→comfyui→mflux | **504 张量逐元素相等** |
| gordonz 美式漫画 | peft + 合并 qkv，328 键 | → mflux 492 键 | 492/492 被 mflux 加载器命中（原文件 246/328） |

第三行是重点：**原文件直接加载只有 246/328 命中**（82 个 qkv 键被丢弃，注意力层的风格完全没生效），转换后 492/492。

---

## 限制

- **只处理 MOSS-Image 的 DiT LoRA**。text encoder 的 LoRA、LoKr/LoHa 等其他低秩格式不在范围内。
- **不修数值问题**。如果 LoRA 本身训练坏了，转换救不了。
- **bf16 需要 MLX**。纯 numpy 后端读 bf16 safetensors 会报错。
- 层号是**位置索引**，不做跨版本重映射。如果 MOSS 未来改了 block 结构，层号会对不上。

---

## 许可证

Apache-2.0。代码不含任何模型权重。

（MOSS-Image 模型本身遵循 OpenMOSS 的许可，与本工具无关。）
