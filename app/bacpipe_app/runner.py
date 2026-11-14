"""
Starting, watching and stopping the pipeline process (worker.py).

``PipelineRun`` starts the worker with the values of the form and collects
everything it prints in an ``OutputBuffer``. The pages read from that buffer,
so a browser tab that is opened (or reloaded) in the middle of a run shows
the complete output so far.
"""

import codecs
import datetime
import os
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import yaml

from . import paths

#: width in characters of the progress bars drawn by tqdm
TERMINAL_COLUMNS = 110


class OutputBuffer:
    """
    Thread safe text buffer with a position counter.

    Readers remember how far they have read (``cursor``) and ask for
    everything after that. Only the last ``max_chars`` characters are kept,
    which is plenty for the terminal view; the complete output is in the log
    file of the run.
    """

    def __init__(self, max_chars=2_000_000):
        self._chunks = deque()
        self._start = 0  # position of the first kept character
        self._end = 0  # position after the last character
        self._max_chars = max_chars
        self._lock = threading.Lock()

    def write(self, text):
        if not text:
            return
        with self._lock:
            self._chunks.append(text)
            self._end += len(text)
            while self._end - self._start > self._max_chars and len(self._chunks) > 1:
                self._start += len(self._chunks.popleft())

    def read_from(self, cursor):
        """Return ``(text written after cursor, new cursor)``."""
        with self._lock:
            if cursor >= self._end:
                return "", self._end
            text = "".join(self._chunks)
            return text[max(cursor - self._start, 0):], self._end


class PipelineRun:
    """
    One run of the pipeline in a worker process.

    Parameters
    ----------
    config : dict
        values for ``bacpipe.config``
    settings : dict
        values for ``bacpipe.settings``
    on_exit : callable, optional
        called with this object (from a background thread) once the process
        has ended
    """

    def __init__(self, config, settings, on_exit=None):
        self.config = config
        self.settings = settings
        self.on_exit = on_exit
        self.output = OutputBuffer()
        self.returncode = None
        self.stopped_by_user = False
        self.started = time.time()
        self.ended = None

        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = paths.state_dir() / "runs" / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_file = self.run_dir / "run.yaml"
        self.log_file = self.run_dir / "terminal_output.log"
        self.dashboard_kwargs_file = self.run_dir / "dashboard_kwargs.pkl"
        self._process = None

    # ----------------------------------------------------------------
    def start(self):
        """Write ``run.yaml`` and start the worker process."""
        self.run_file.write_text(
            yaml.safe_dump(
                {
                    "config": self.config,
                    "settings": self.settings,
                    "dashboard_kwargs_file": str(self.dashboard_kwargs_file),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        popen_kwargs = paths.child_popen_kwargs()
        if os.name != "nt":
            # own process group, so that stop() reaches the helper processes
            # of the pipeline as well
            popen_kwargs["start_new_session"] = True

        self._process = subprocess.Popen(
            paths.child_command("worker.py", str(self.run_file)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # one stream, like in a terminal
            stdin=subprocess.DEVNULL,
            # relative paths behave like in `python run_pipeline.py`, with
            # the workspace taking the place of the repository folder
            cwd=str(paths.workspace_dir()),
            env=paths.child_env(
                COLUMNS=TERMINAL_COLUMNS,
                TQDM_NCOLS=TERMINAL_COLUMNS,
            ),
            **popen_kwargs,
        )
        threading.Thread(target=self._collect_output, daemon=True).start()

    def _collect_output(self):
        """Background thread: copy the output of the worker into the buffer."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with open(self.log_file, "w", encoding="utf-8", newline="") as log:
            while True:
                data = self._process.stdout.read1(65536)
                if not data:
                    break
                text = decoder.decode(data)
                self.output.write(text)
                log.write(text)
                log.flush()
            returncode = self._process.wait()

            if self.stopped_by_user:
                closing = "\n\n--- stopped ---\n"
            elif returncode != 0:
                closing = (
                    f"\n\n--- the pipeline ended with an error (exit code "
                    f"{returncode}) ---\nThe complete output was saved to "
                    f"{self.log_file}\n"
                )
            else:
                closing = ""
            self.output.write(closing)
            log.write(closing)

        self.ended = time.time()
        self.returncode = returncode
        if self.on_exit is not None:
            self.on_exit(self)

    # ----------------------------------------------------------------
    @property
    def running(self):
        return self.returncode is None

    @property
    def succeeded(self):
        return self.returncode == 0 and not self.stopped_by_user

    @property
    def wants_dashboard(self):
        """True if the pipeline got to the point of opening the dashboard."""
        return self.succeeded and self.dashboard_kwargs_file.exists()

    def elapsed(self):
        return (self.ended or time.time()) - self.started

    def stop(self):
        """Stop the worker and every process it started."""
        process = self._process
        if process is None or process.poll() is not None:
            return
        self.stopped_by_user = True
        if os.name == "nt":
            # /T = including child processes, /F = force
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                **paths.child_popen_kwargs(),
            )
        else:
            # like Ctrl+C in a terminal first, bacpipe cleans up on that
            _signal_group(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _signal_group(process.pid, signal.SIGKILL)


def _signal_group(pid, sig):
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def absolute_path_values(values, keys):
    """
    Make the folder values absolute by resolving them against the workspace.

    Parameters
    ----------
    values : dict
        values of a form
    keys : iterable of str
        keys holding folders (the fields of kind "folder")

    Returns
    -------
    dict
        copy of ``values`` with absolute folders
    """
    result = dict(values)
    for key in keys:
        value = result.get(key)
        if not isinstance(value, str) or not value:
            continue
        if value == paths.TEST_DATA_AUDIO_DIR:
            continue  # special value understood by bacpipe.play
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = paths.workspace_dir() / path
        result[key] = str(path)
    return result
