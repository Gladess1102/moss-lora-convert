"""ComfyUI 自定义节点：把 LoRA 转成 ComfyUI / mflux 能吃的格式。

安装：把整个 moss-lora-convert 目录放到 ComfyUI/custom_nodes/ 下（或做软链接），
节点会出现在 MOSS/LoRA 分类里。

依赖：safetensors（ComfyUI 自带）。有 mlx 时自动用它处理 bf16。
"""
import os
import sys

# 允许作为 custom_nodes 子目录直接加载（父目录里有 mosslora 包）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_PARENT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import folder_paths
except ImportError:  # 不在 ComfyUI 里（比如单测）
    folder_paths = None

from mosslora.convert import convert, get_backend
from mosslora.formats import detect


def _loras_dir():
    if folder_paths:
        try:
            return folder_paths.get_folder_paths("loras")[0]
        except Exception:
            pass
    return os.path.join(os.getcwd(), "models", "loras")


def _list_loras():
    if folder_paths:
        try:
            names = folder_paths.get_filename_list("loras")
            if names:
                return names
        except Exception:
            pass
    d = _loras_dir()
    out = []
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith((".safetensors", ".ckpt")):
                    out.append(os.path.relpath(os.path.join(root, f), d))
    return sorted(out) or ["<把 LoRA 放进 models/loras>"]


class MOSSLoRAConvert:
    """把一个 LoRA 文件转成目标格式，返回新文件的绝对路径。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (_list_loras(), {"tooltip": "models/loras 里的文件"}),
                "target": (["comfyui", "mflux", "kohya", "peft"], {"default": "comfyui"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("converted_path",)
    FUNCTION = "run"
    CATEGORY = "MOSS/LoRA"
    OUTPUT_NODE = False

    def run(self, lora_name, target, overwrite):
        if lora_name.startswith("<"):
            raise RuntimeError("models/loras 里没有 LoRA 文件")
        src = lora_name
        if folder_paths:
            try:
                src = folder_paths.get_full_path("loras", lora_name)
            except Exception:
                src = os.path.join(_loras_dir(), lora_name)
        if not src or not os.path.isfile(src):
            raise RuntimeError("找不到文件: %s" % lora_name)

        base, ext = os.path.splitext(src)
        dst = "%s.%s%s" % (base, target, ext)
        if os.path.exists(dst) and not overwrite:
            print("[moss-lora-convert] 已存在，跳过: %s" % dst)
            return (dst,)

        stats = convert(src, dst, to=target, verbose=True)
        print("[moss-lora-convert] %s -> %s (%d 键)"
              % (os.path.basename(src), os.path.basename(dst), stats["dst_keys"]))
        return (dst,)


class MOSSLoRAInspect:
    """只检测格式，不写文件。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"lora_name": (_list_loras(),)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "MOSS/LoRA"

    def run(self, lora_name):
        src = lora_name
        if folder_paths:
            try:
                src = folder_paths.get_full_path("loras", lora_name)
            except Exception:
                src = os.path.join(_loras_dir(), lora_name)
        be = get_backend()
        w = be.load(src)
        info = detect(w.keys())
        projs = sorted({k.split(".")[-3] for k in w if ".proj." in k})
        lines = [
            "文件: %s" % os.path.basename(src),
            "键数: %d" % len(w),
            "风格: %s" % info["style"],
            "命名: %s" % info["lora_naming"],
            "投影: %s" % ", ".join(projs),
        ]
        report = "\n".join(lines)
        print(report)
        return (report,)


NODE_CLASS_MAPPINGS = {
    "MOSSLoRAConvert": MOSSLoRAConvert,
    "MOSSLoRAInspect": MOSSLoRAInspect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSLoRAConvert": "MOSS LoRA Convert",
    "MOSSLoRAInspect": "MOSS LoRA Inspect",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
