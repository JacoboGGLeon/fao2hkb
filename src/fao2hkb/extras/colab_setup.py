# %%capture
from __future__ import annotations

TORCH_CU121_ARGS = [
    "--upgrade",
    "--index-url", "https://download.pytorch.org/whl/cu121",
    "torch",
]

BASE_PACKAGES = [
    # Core DS (ojo: si ya están importados, NO se actualizan aquí)
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "tqdm",

    # HF / embeddings
    "sentence-transformers",
    "transformers>=4.40,<5",
    "huggingface_hub>=0.23.0",
    "accelerate",
    "safetensors",

    # IO / Drive helpers
    "gdown",
    "google-api-python-client",
    "google-auth",
    "google-auth-httplib2",
    "google-auth-oauthlib",

    # Infra útil para robustez
    "requests",
    "tenacity",

    # Pydantic 2
    "pydantic>=2,<3",
]

import os, sys, json, platform, subprocess, importlib.util
from pathlib import Path
from datetime import datetime

def _run(cmd: list[str], check: bool = True) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        if check:
            raise
        return f"<error: {e!r}>"

def pipi(args: list[str]) -> None:
    print(f"📦 pip install {' '.join(args)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

def load_hf_token_from_colab() -> None:
    try:
        from google.colab import userdata  # type: ignore
        hf_token = userdata.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            print("🔐 HF_TOKEN cargado desde Colab Secrets.")
        else:
            print("ℹ️ HF_TOKEN no definido; usarás solo modelos públicos.")
    except Exception:
        pass

def _bootstrap_env_dir() -> Path:
    d = Path.cwd() / "_bootstrap_env"
    d.mkdir(parents=True, exist_ok=True)
    return d

def record_bootstrap_env() -> None:
    out_dir = _bootstrap_env_dir()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    env_json = out_dir / f"env_{stamp}.json"
    freeze_txt = out_dir / f"pip_freeze_{stamp}.txt"

    info = {
        "timestamp_utc": stamp,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "pip_version": _run([sys.executable, "-m", "pip", "--version"], check=False),
        "nvidia_smi": _run(["bash", "-lc", "nvidia-smi -L"], check=False),
    }

    try:
        freeze = _run([sys.executable, "-m", "pip", "freeze"], check=False)
        freeze_txt.write_text(freeze + "\n", encoding="utf-8")
    except Exception as e:
        freeze_txt.write_text(f"<error: {e!r}>\n", encoding="utf-8")

    env_json.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧾 Bootstrap env guardado: {env_json.name} + {freeze_txt.name}")

def _modules_already_loaded() -> set[str]:
    # Si alguno está cargado, NO conviene tocarlo con pip --upgrade sin restart.
    watched = {"numpy", "pandas", "sklearn", "matplotlib", "torch"}
    return {m for m in watched if m in sys.modules}

def _write_constraints(path: str, keys: list[str]) -> None:
    # OJO: usamos importlib.metadata (no importa el módulo).
    try:
        from importlib import metadata
    except Exception:
        return

    lines = []
    for k in keys:
        try:
            v = metadata.version(k)
            lines.append(f"{k}=={v}")
        except Exception:
            pass

    if not lines:
        return

    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"🧷 constraints guardado: {path}")

def maybe_install_torch_cu121() -> None:
    # Si torch ya fue importado, no lo toques aquí si quieres evitar restart.
    if "torch" in sys.modules:
        print("ℹ️ torch ya está importado en este kernel → NO reinstalo torch (evita mismatch sin restart).")
        return

    try:
        import importlib
        # no importamos torch, solo detectamos si está instalado
        from importlib import metadata
        _ = metadata.version("torch")
        torch_installed = True
    except Exception:
        torch_installed = False

    # Si está instalado pero quieres forzar cu121 igual, necesitarías importarlo para ver cuda,
    # pero eso rompe el objetivo de 'no tocar torch ya cargado'. En runtime fresco sí conviene.
    if not torch_installed:
        print("ℹ️ torch no detectado; instalando cu121…")
        pipi(TORCH_CU121_ARGS)
    else:
        # Runtime fresco: instalamos cu121 de todas formas para alinear (sin haber importado torch).
        print("ℹ️ torch detectado (sin importar). Alineando a cu121 (best-effort)…")
        pipi(TORCH_CU121_ARGS)

def install_base_packages_no_restart() -> None:
    loaded = _modules_already_loaded()
    if loaded:
        print(f"⚠️ Ya hay módulos binarios cargados: {sorted(loaded)}")
        print("   Para evitar restart, NO voy a actualizar numpy/pandas/sklearn/matplotlib/seaborn/torch en este kernel.")

    # Congelamos versiones ya instaladas para que pip no intente subir numpy por deps.
    constraints_path = str(_bootstrap_env_dir() / "constraints_no_restart.txt")
    _write_constraints(constraints_path, keys=["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "torch"])

    skip = {"numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "torch"}
    pkgs = []
    for spec in BASE_PACKAGES:
        name = spec.split("==")[0].split(">=")[0].split("<")[0].strip()
        if name in skip and (name in {"numpy","pandas","matplotlib","seaborn"} and ("numpy" in loaded or "pandas" in loaded or "matplotlib" in loaded)):
            continue
        if name in skip and (name in {"scikit-learn"} and ("sklearn" in loaded)):
            continue
        if name in skip and (name == "torch" and ("torch" in loaded)):
            continue
        pkgs.append(spec)

    if not pkgs:
        print("ℹ️ No hay paquetes que instalar sin tocar binarios ya cargados.")
        return

    print("🚀 Instalando paquetes base (modo no-restart)…")
    pipi(["--upgrade", "--no-cache-dir", "--upgrade-strategy", "only-if-needed", "-c", constraints_path, *pkgs])

def setup_flash_attention2_and_runtime() -> None:
    # Solo si torch no está roto/importable. Si torch ya importado y está bien, ok.
    try:
        import torch
    except Exception as e:
        print("⚠️ torch no importable ahora mismo:", repr(e))
        print("   Si acabas de instalar torch/numpy, esto suele requerir restart.")
        return

    user_enable = os.environ.get("ENABLE_FA2", "").strip()
    forced = (user_enable == "1") if user_enable in {"0", "1"} else None

    if not torch.cuda.is_available():
        print("⚠️ No hay GPU CUDA disponible. Se usará CPU / SDPA.")
        os.environ["ENABLE_FA2"] = "0"
        os.environ["FA2_AVAILABLE"] = "0"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        return

    gpu_name = torch.cuda.get_device_name(0)
    try:
        sm = torch.cuda.get_device_capability(0)
    except Exception:
        sm = (0, 0)
    sm_num = sm[0] * 10 + sm[1]

    is_l4 = "L4" in gpu_name.upper()
    allow_fa2_on_l4 = os.environ.get("ALLOW_FA2_ON_L4", "0") == "1"
    auto_enable = (sm_num >= 80) and (not is_l4 or allow_fa2_on_l4)

    if forced is False:
        enable_fa2 = False
    elif forced is True:
        enable_fa2 = True
    else:
        enable_fa2 = auto_enable

    os.environ["ENABLE_FA2"] = "1" if enable_fa2 else "0"

    if enable_fa2:
        try:
            if importlib.util.find_spec("flash_attn") is None:
                pipi(["--no-build-isolation", "flash-attn"])
            import flash_attn  # noqa
            os.environ["FA2_AVAILABLE"] = "1"
            print("✅ FlashAttention-2 disponible.")
        except Exception as e:
            os.environ["FA2_AVAILABLE"] = "0"
            os.environ["ENABLE_FA2"] = "0"
            print(f"⚠️ FlashAttention-2 no usable ({e}). Se usará SDPA por defecto.")
    else:
        os.environ["FA2_AVAILABLE"] = "0"
        print("ℹ️ FA2 desactivado. Se usará SDPA por defecto.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    print(
        f"torch={torch.__version__} · CUDA={torch.version.cuda} · "
        f"cuda.is_available={torch.cuda.is_available()} · "
        f"ENABLE_FA2={os.environ.get('ENABLE_FA2')} · FA2_AVAILABLE={os.environ.get('FA2_AVAILABLE')}"
    )

# ----------------------------
# Ejecución (no-restart)
# ----------------------------
load_hf_token_from_colab()

# NO recomiendo upgrade de pip/setuptools/wheel si tu meta es "cero restarts".
# (en Colab a veces mete cambios raros)

def colab_setup() -> None:
    """Run the original Colab dependency bootstrap (from the notebook), on-demand."""
    maybe_install_torch_cu121()
    install_base_packages_no_restart()
    setup_flash_attention2_and_runtime()
    try:
        record_bootstrap_env()
    except Exception:
        pass

if __name__ == "__main__":
    colab_setup()
