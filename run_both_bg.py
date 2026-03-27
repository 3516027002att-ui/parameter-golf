import os
import subprocess
import sys
import time

venv_python = r"D:\Repository\parameter-golf\.venv\Scripts\python.exe"

os.environ["DATA_PATH"] = "./data/datasets/fineweb10B_sp1024"
os.environ["TOKENIZER_PATH"] = "./data/tokenizers/fineweb_1024_bpe.model"
os.environ["TRAIN_BATCH_TOKENS"] = "8192"
os.environ["MAX_WALLCLOCK_SECONDS"] = "999999"


def run(name, script):
    os.environ["RUN_ID"] = name
    log_file = f"logs/local_{name}.txt"
    print(f"\n{'=' * 60}")
    print(f"Starting {script} (log: {log_file})")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            [venv_python, script],
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    with open(log_file + ".pid", "w") as f:
        f.write(str(proc.pid))

    print(f"Started with PID {proc.pid}")
    print(f"Monitor with: Get-Content logs/local_{name}.txt -Tail 20")
    sys.stdout.flush()

    proc.wait()

    print(f"\n{'=' * 60}")
    print(f"Completed {script} with exit code {proc.returncode}")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()
    return proc.returncode


if __name__ == "__main__":
    print("Starting training runs...")
    sys.stdout.flush()
    rc1 = run("traingpt_full1", "train_gpt.py")
    if rc1 == 0:
        rc2 = run("plan1_full3", "plan1.py")
