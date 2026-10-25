import sys
import time
import pytest
import socket
import launcher
import subprocess
from pathlib import Path

LAUNCHER_DIR = str(Path(launcher.__file__).parent)


def test_read_env_handles_quotes_spaces_comments(tmp_path):
    p = tmp_path / ".env"
    p.write_text('# c\nA = "x y"\nB=2\nbad line\n', encoding="utf-8")
    assert launcher.read_env(p) == {"A": "x y", "B": "2"}


def test_find_free_port_skips_busy_and_excluded():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert launcher.find_free_port(busy, exclude={busy + 1}) not in (busy, busy + 1)


def test_needs_install_tracks_requirements_hash(tmp_path):
    req, venv = tmp_path / "requirements.txt", tmp_path / ".venv"
    venv.mkdir()
    req.write_text("a==1")
    assert launcher.needs_install(venv, req)
    (venv / ".installed").write_text(launcher.requirements_hash(req))
    assert not launcher.needs_install(venv, req)
    req.write_text("a==2")
    assert launcher.needs_install(venv, req)


def test_wait_for_stops_when_child_dies():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    t = time.monotonic()
    assert launcher.wait_for(lambda: False, [p], timeout=10) is False
    assert time.monotonic() - t < 2


def test_wait_for_success_and_timeout():
    assert launcher.wait_for(lambda: True, [], timeout=1)
    assert launcher.wait_for(lambda: False, [], timeout=0.6) is False


def test_stop_kills_process():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    launcher.stop([p])
    assert p.poll() is not None


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")
def test_job_object_kills_children_when_owner_is_killed(tmp_path):
    pidfile = tmp_path / "pid"
    owner = tmp_path / "owner.py"
    owner.write_text(
        "import sys, subprocess, time\n"
        f"sys.path.insert(0, {LAUNCHER_DIR!r})\n"
        "import launcher\n"
        "job = launcher.KillOnCloseJob()\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "job.add(c)\n"
        f"open({str(pidfile)!r}, 'w').write(str(c.pid))\n"
        "time.sleep(60)\n"
    )
    o = subprocess.Popen([sys.executable, str(owner)])
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text():
            break
        time.sleep(0.1)
    child = pidfile.read_text()
    o.kill()  # hard kill: no cleanup code runs, only the Job Object can stop the child
    o.wait()
    time.sleep(1)
    out = subprocess.run(["tasklist", "/FI", f"PID eq {child}"], capture_output=True, text=True).stdout
    assert child not in out
