"""VM-side launcher for the SRD long-context run (detached, idempotent)."""
import subprocess
subprocess.Popen(
    "cd /content && chmod +x colab_srdlc_run.sh && "
    "nohup bash colab_srdlc_run.sh >> /content/srd_lc.log 2>&1 & echo $!",
    shell=True, executable="/bin/bash")
print("LAUNCHED", flush=True)
