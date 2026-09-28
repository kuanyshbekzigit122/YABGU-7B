import sys
import subprocess
import time

def run_with_retry(script_path, timeout=900, max_retries=3):
    cmd = [
        "colab", "exec",
        "-s", "yabgu",
        "--timeout", str(float(timeout)),
        "-f", script_path
    ]
    for attempt in range(1, max_retries + 1):
        print(f"\n[RUNNER] Attempt {attempt}/{max_retries}: Running '{script_path}' on Colab session 'yabgu'...")
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print(f"[RUNNER] Successfully finished '{script_path}' (Exit Code: 0)")
            return 0
        print(f"[RUNNER] Execution exited with code {res.returncode}. Waiting 5 seconds before retry...")
        time.sleep(5)
    print(f"[RUNNER] Failed after {max_retries} attempts.")
    return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/colab_run.py <script_to_run> [timeout_seconds]")
        sys.exit(1)
    target = sys.argv[1]
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    sys.exit(run_with_retry(target, timeout=t))
