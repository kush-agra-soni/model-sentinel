"""One-click launcher: venv + install + backend + frontend + browser; closing it stops everything."""
import os
import sys
import time
import shutil
import socket
import hashlib
import subprocess
import webbrowser
import urllib.request
from pathlib import Path

APP_NAME = "Tabular ML Platform"
DEFAULT_BACKEND_PORT = 8005
DEFAULT_FRONTEND_PORT = 8505
READY_TIMEOUT = 120
MIN_PY = (3, 10)


def repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def read_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_free_port(start: int, exclude: set[int] = frozenset(), tries: int = 50) -> int:
    for p in range(start, start + tries):
        if p not in exclude and port_free(p):
            return p
    raise RuntimeError(f"No free port found in {start}-{start + tries}")


def requirements_hash(req: Path) -> str:
    return hashlib.sha256(req.read_bytes()).hexdigest()


def needs_install(venv: Path, req: Path) -> bool:
    marker = venv / ".installed"
    return not marker.exists() or marker.read_text().strip() != requirements_hash(req)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def find_system_python() -> list[str] | None:
    candidates = [["py", "-3.12"], ["py", "-3.11"], ["py", "-3"], ["python3"], ["python"]]
    for cmd in candidates:
        if not shutil.which(cmd[0]):
            continue
        try:
            out = subprocess.run(
                cmd + ["-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                capture_output=True, text=True, timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0:
            major, minor = map(int, out.stdout.split())
            if (major, minor) >= MIN_PY:
                return cmd
    return None


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def wait_for(check, procs: list[subprocess.Popen], timeout: float = READY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in procs):
            return False
        if check():
            return True
        time.sleep(0.5)
    return False


class KillOnCloseJob:
    """Windows Job Object: every process assigned (and its children) dies when this launcher exits."""

    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self._k32 = k32
        self.handle = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.handle = None

    def add(self, proc: subprocess.Popen) -> None:
        if self.handle:
            self._k32.AssignProcessToJobObject(self.handle, int(proc._handle))


def exit_when_bootloader_dies() -> None:
    """A PyInstaller one-file exe runs us in a child of its bootloader; if the bootloader is
    killed (e.g. Task Manager) we must exit too so the Job Object closes and kills the services."""
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    import ctypes
    import threading
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                               ctypes.POINTER(wintypes.DWORD)]
    k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    handle = k32.OpenProcess(0x00100000 | 0x1000, False, os.getppid())  # SYNCHRONIZE | QUERY_LIMITED
    if not handle:
        return
    buf, size = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
    if not k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
        return
    if Path(buf.value).resolve() != Path(sys.executable).resolve():
        return

    def watch() -> None:
        k32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            else:
                p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def fail(msg: str) -> None:
    print(f"\n[ERROR] {msg}")
    if getattr(sys, "frozen", False):
        input("Press Enter to close...")
    sys.exit(1)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    exit_when_bootloader_dies()
    root = repo_root()
    os.chdir(root)
    print(f"=== {APP_NAME} launcher ===\nFolder: {root}\n")
    req, venv = root / "requirements.txt", root / ".venv"

    if not (root / ".env").exists() and (root / ".env.example").exists():
        shutil.copy(root / ".env.example", root / ".env")
        print("Created .env from .env.example - edit LLM_BASE_URL / LLM_API_KEY if needed.")
    env_file = read_env(root / ".env")

    py = venv_python(venv)
    if not py.exists():
        sys_py = find_system_python()
        if not sys_py:
            fail("Python 3.10+ not found. Install it from https://www.python.org/downloads/ "
                 "(tick 'Add python.exe to PATH') and run the launcher again.")
        print(f"[1/4] Creating virtual environment with {' '.join(sys_py)} ...")
        if subprocess.run(sys_py + ["-m", "venv", str(venv)]).returncode != 0:
            fail("Could not create the virtual environment.")
    else:
        print("[1/4] Virtual environment found.")

    if needs_install(venv, req):
        print("[2/4] Installing requirements (first run takes a few minutes) ...")
        if subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True).returncode != 0:
            subprocess.run([str(py), "-m", "ensurepip", "--upgrade"], capture_output=True)
        cmds = [[str(py), "-m", "pip", "install", "--upgrade", "pip", "-q"],
                [str(py), "-m", "pip", "install", "-r", str(req)]]
        for cmd in cmds:
            if subprocess.run(cmd).returncode != 0:
                fail("Dependency installation failed - see the output above.")
        (venv / ".installed").write_text(requirements_hash(req))
    else:
        print("[2/4] Requirements already installed.")

    bport = find_free_port(int(env_file.get("BACKEND_PORT", DEFAULT_BACKEND_PORT)))
    fport = find_free_port(int(env_file.get("FRONTEND_PORT", DEFAULT_FRONTEND_PORT)), {bport})
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    child_env = {**os.environ, "BACKEND_PORT": str(bport), "BACKEND_URL": f"http://127.0.0.1:{bport}",
                 "PYTHONUNBUFFERED": "1"}

    print(f"[3/4] Starting backend on :{bport} and frontend on :{fport} ...")
    job = KillOnCloseJob()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    blog = open(logs / "backend.log", "w", encoding="utf-8")
    flog = open(logs / "frontend.log", "w", encoding="utf-8")
    procs = [
        subprocess.Popen(
            [str(py), "-m", "uvicorn", "--factory", "sentinel.api:create_app", "--app-dir", "src",
             "--host", "127.0.0.1", "--port", str(bport)],
            stdout=blog, stderr=subprocess.STDOUT, env=child_env, creationflags=flags),
        subprocess.Popen(
            [str(py), "-m", "streamlit", "run", "ui/app.py", "--server.port", str(fport),
             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
            stdout=flog, stderr=subprocess.STDOUT, env=child_env, creationflags=flags),
    ]
    for p in procs:
        job.add(p)

    try:
        ok = wait_for(lambda: http_ok(f"http://127.0.0.1:{bport}/health")
                      and http_ok(f"http://127.0.0.1:{fport}/_stcore/health"), procs)
        if not ok:
            stop(procs)
            fail("App did not start. Check logs/backend.log and logs/frontend.log.")
        url = f"http://localhost:{fport}"
        print(f"[4/4] Ready -> {url}\n\nClose this window or press Ctrl+C to stop the app.")
        webbrowser.open(url)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("\nA service exited unexpectedly - see logs/. Shutting down.")
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        stop(procs)
        blog.close()
        flog.close()


if __name__ == "__main__":
    main()
