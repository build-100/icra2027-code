"""Read-only Windows resource telemetry and a cooperative training gate."""
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


class Memory(C.Structure):
    _fields_ = [("length", W.DWORD), ("load", W.DWORD)] + [(name, C.c_ulonglong) for name in
        ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]


def system_times():
    idle, kernel, user = W.FILETIME(), W.FILETIME(), W.FILETIME()
    if not C.windll.kernel32.GetSystemTimes(C.byref(idle), C.byref(kernel), C.byref(user)):
        raise C.WinError()
    return tuple((v.dwHighDateTime << 32) + v.dwLowDateTime for v in (idle, kernel, user))


class ResourceMonitor:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.previous = system_times()
        self.latest = None
        self.error = None
        self.pauses = 0

    def sample(self):
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"], text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        name, temp, load, used, total, power = [part.strip() for part in raw.strip().splitlines()[0].split(",")]
        memory = Memory()
        memory.length = C.sizeof(memory)
        if not C.windll.kernel32.GlobalMemoryStatusEx(C.byref(memory)):
            raise C.WinError()
        current = system_times()
        idle, kernel, user = [a - b for a, b in zip(current, self.previous)]
        denominator = kernel + user
        self.previous = current
        cpu_load = 100 * (1 - idle / denominator) if denominator else 0.0
        row = {"time": time.time(), "gpu_name": name, "gpu_temperature_c": float(temp),
               "gpu_utilization_percent": float(load), "gpu_memory_used_mib": float(used),
               "gpu_memory_total_mib": float(total), "gpu_power_w": float(power),
               "cpu_load_percent": cpu_load, "logical_cpus": os.cpu_count(),
               "ram_total_gib": memory.total / 2**30, "ram_available_gib": memory.available / 2**30,
               "cpu_temperature_c": None, "cpu_temperature_status": "vendor_WMI_access_denied_not_measured"}
        self.latest, self.error = row, None
        append(self.directory / "resources.jsonl", row)
        dump(self.directory / "resources_latest.json", row)
        return row

    def _loop(self):
        while not self.stop_event.wait(5):
            try:
                self.sample()
            except Exception as error:
                self.error = repr(error)
                append(self.directory / "telemetry_errors.jsonl", {"time": time.time(), "error": self.error})

    def start(self):
        row = self.sample()
        if row["ram_available_gib"] < 20 or row["gpu_temperature_c"] >= 80 or row["gpu_memory_used_mib"] > .8 * row["gpu_memory_total_mib"]:
            raise RuntimeError("Insufficient resource headroom to start this run.")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def gate(self):
        paused = False
        while True:
            row = self.latest
            if row is None or time.time() - row["time"] > 30:
                reason = "stale_resource_telemetry"
            elif row["gpu_temperature_c"] >= (78 if paused else 85):
                reason = "gpu_temperature"
            elif row["ram_available_gib"] < (18 if paused else 12):
                reason = "available_RAM"
            elif row["gpu_memory_used_mib"] > (.8 if paused else .9) * row["gpu_memory_total_mib"]:
                reason = "VRAM_headroom"
            else:
                if paused:
                    append(self.directory / "scheduling.jsonl", {"time": time.time(), "action": "resume", "sample": row})
                return
            if not paused:
                self.pauses += 1
                append(self.directory / "scheduling.jsonl", {"time": time.time(), "action": "pause", "reason": reason, "sample": row})
                paused = True
            time.sleep(5)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=12)


def reserve_cpu_headroom():
    kernel = C.windll.kernel32
    kernel.GetCurrentProcess.restype = W.HANDLE
    handle = kernel.GetCurrentProcess()
    kernel.SetPriorityClass.argtypes = [W.HANDLE, W.DWORD]
    kernel.SetProcessAffinityMask.argtypes = [W.HANDLE, C.c_size_t]
    count = os.cpu_count() or 1
    reserved = min(4, max(0, count - 1))
    mask = ((1 << count) - 1) ^ ((1 << reserved) - 1)
    if not kernel.SetPriorityClass(handle, 0x4000):
        raise C.WinError()
    if not kernel.SetProcessAffinityMask(handle, mask):
        raise C.WinError()
    return {"priority": "BelowNormal", "reserved_logical_cpus": reserved, "affinity_mask": mask}
