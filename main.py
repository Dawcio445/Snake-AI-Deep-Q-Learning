from __future__ import annotations

import os
import subprocess
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from config import CFG
from global_runner import run_global
from train_single import train


def maybe_launch_detached_cmd() -> bool:
    if os.name != "nt":
        return False
    if "--detached-child" in sys.argv:
        return False

    script_path = Path(__file__).resolve()
    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    cmd = [sys.executable, str(script_path), "--detached-child", *sys.argv[1:]]
    try:
        subprocess.Popen(cmd, creationflags=creation_flags, close_fds=True)
    except Exception as exc:
        print(f"Nie udało się otworzyć nowego okna: {exc}")
        return False
    return True


def main() -> None:
    if maybe_launch_detached_cmd():
        sys.exit(0)
    if CFG.mode == "single":
        train(CFG)
    elif CFG.mode == "global":
        run_global(CFG)
    else:
        raise ValueError("CFG.mode musi być 'single' albo 'global'.")

import multiprocessing as mp
import torch.multiprocessing as tmp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    tmp.set_sharing_strategy("file_system")
    main()
