import atexit
import platform
import subprocess
import threading


def _clear_cache_command() -> list[str]:
    """The command that drops the OS page cache, per platform.

    `sh -c` is required: the exit status is the last command's, so a failed
    purge/drop_caches is visible in the return code.
    """
    system = platform.system()
    if system == "Darwin":
        return ["sudo", "-n", "sh", "-c", "sync; purge"]
    elif system == "Linux":
        return ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
    else:
        raise Exception("Unsupported platform")


def ensure_sudo():
    """Authenticate once, then keep the sudo timestamp fresh for the whole run.

    Only cache clearing needs root. Benchmark processes run as the normal user
    so that their measured RSS is not distorted.
    """
    if subprocess.call(["sudo", "-v"]) != 0:
        raise RuntimeError(
            "sudo authentication failed. Cache clearing requires root, and "
            "measurements taken on a warm cache would be invalid."
        )

    stop = threading.Event()

    def refresh():
        while not stop.wait(60):
            subprocess.call(
                ["sudo", "-n", "-v"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    threading.Thread(target=refresh, daemon=True).start()
    atexit.register(stop.set)


def clear_cache():
    """Drop the OS page cache, or raise if it could not be dropped."""
    result = subprocess.run(_clear_cache_command(), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to clear the page cache (exit {result.returncode}). "
            f"Measurements would be warm-cache and invalid. "
            f"stderr: {result.stderr.strip()}"
        )


def time_args():
    system = platform.system()
    if system == "Darwin":
        return ["gtime", "-v"]
    elif system == "Linux":
        return ["time", "-v"]
    else:
        raise Exception("Unsupported platform")
