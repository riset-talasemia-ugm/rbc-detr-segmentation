"""Hentikan lebih awal bila torch yang terpasang tidak punya kernel untuk GPU ini (mis. RTX 50-series/Blackwell butuh torch CUDA 12.8+)."""
import sys


def arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """True bila torch dikompilasi untuk arsitektur GPU ini (sm_XY). PTX kompatibel-maju sengaja tidak diandalkan:
    pada GPU Blackwell dengan torch CUDA 12.4 laporan pengguna tetap berakhir dengan "no kernel image"."""
    return f"sm_{capability[0]}{capability[1]}" in arch_list


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("torch tidak melihat GPU (torch.cuda.is_available() False). Periksa build torch (harus build CUDA) dan driver.")
    cap, name = torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0)
    archs = torch.cuda.get_arch_list()
    print(f"GPU: {name} (sm_{cap[0]}{cap[1]}); torch {torch.__version__} (CUDA {torch.version.cuda}) mendukung: {' '.join(archs)}")
    if not arch_supported(cap, archs):
        sys.exit(
            f"torch ini tidak punya kernel untuk sm_{cap[0]}{cap[1]} ({name}). Training akan gagal dengan 'no kernel image'.\n"
            "GPU Blackwell (RTX 50-series) butuh torch build CUDA 12.8+. Perbaiki lalu jalankan ulang:\n"
            "  uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
            "Untuk melewati pemeriksaan ini (mis. false negative): SKIP_ARCH_CHECK=1 bash run_vast.sh"
        )


if __name__ == "__main__":
    main()
