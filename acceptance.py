#!/usr/bin/env python3
"""moss-lora-convert 验收：三份真实社区 LoRA 的往返与加载。"""
import sys, os, subprocess
sys.path.insert(0, "/Users/yangyuxin/moss-image/moss-lora-convert")
from mosslora import convert, roundtrip_check, get_backend

ROOT = "/Users/yangyuxin/moss-image"
BE = get_backend("mlx")
ok_all = True

def check(name, fn):
    global ok_all
    try:
        ok, msg = fn()
    except Exception as e:
        ok, msg = False, "%s: %s" % (type(e).__name__, e)
    print("%-28s %s  %s" % (name, "PASS" if ok else "FAIL", msg))
    ok_all = ok_all and ok

print("=== 1. akhaliq 4步（mflux, 756 键, 含 alpha + ffn 融合）")
def t1():
    convert(ROOT + "/models/akhaliq/akk_4step_lora.safetensors", "/tmp/acc_akk.comfy.safetensors", to="comfyui")
    convert("/tmp/acc_akk.comfy.safetensors", "/tmp/acc_akk.back.safetensors", to="mflux")
    return roundtrip_check(ROOT + "/models/akhaliq/akk_4step_lora.safetensors", "/tmp/acc_akk.back.safetensors", BE)
check("mflux -> comfyui -> mflux", t1)

print("=== 2. axiomofmind 动画风（mflux, 504 键）")
def t2():
    convert(ROOT + "/models/loras/animated_moss.safetensors", "/tmp/acc_ani.kohya.safetensors", to="kohya")
    convert("/tmp/acc_ani.kohya.safetensors", "/tmp/acc_ani.back.safetensors", to="mflux")
    return roundtrip_check(ROOT + "/models/loras/animated_moss.safetensors", "/tmp/acc_ani.back.safetensors", BE)
check("mflux -> kohya -> mflux", t2)

print("=== 3. gordonz 美式漫画（peft + 合并 qkv, 328 键）")
def t3():
    st = convert(ROOT + "/tmp/lora_dl/american_comic.safetensors", "/tmp/acc_comic.mflux.safetensors", to="mflux")
    if st["dst_keys"] != 492:
        return False, "键数 %d，应为 492" % st["dst_keys"]
    # 再用 mflux 加载器验证命中数
    code = '''
import sys; sys.path.insert(0, "web"); sys.path.insert(0, "venv-mflux024/lib/python3.13/site-packages")
import mflux024_adapter as A
dit = A.load_transformer("models/mlx_vi_8bit")
A.apply_lora(dit, ["/tmp/acc_comic.mflux.safetensors"], [1.0])
'''
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    hit = [l for l in out.splitlines() if "keys matched" in l]
    if not hit:
        return False, "加载输出里没有命中数: " + out[-200:]
    n = hit[0].split("(")[-1].split(" ")[0]
    return (n == "492/492"), "加载命中 " + n
check("peft+qkv -> mflux -> 加载", t3)

print("=== 4. 无损性：comfyui 不融合 down 不同的 qkv")
def t4():
    from mosslora.convert import get_backend as gb, to_ir, normalize_ir, from_ir
    be = gb("mlx")
    w = be.load(ROOT + "/models/akhaliq/akk_4step_lora.safetensors")
    ir, al, unk = to_ir(w, be)
    ir, warn, al = normalize_ir(ir, be, None, al)
    out = from_ir(ir, "comfyui", be, al)
    qkv = [k for k in out if "_proj_qkv." in k]
    return (len(qkv) == 0), "qkv 融合键 %d 个（应为 0，因为 down 不同）" % len(qkv)
check("qkv 融合保护", t4)

print()
print("总判定:", "全部通过" if ok_all else "有失败项")
sys.exit(0 if ok_all else 1)
