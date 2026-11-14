"""
Starting the application: splash screen, web server, browser window.

What happens on start (``main``):

1. If the application is already running, its page is opened in the browser
   and this second process ends.
2. The splash screen is shown (splash.py, own process).
3. The Panel web server is started on a free port. It only listens on
   127.0.0.1, i.e. it is not reachable from other computers and does not
   trigger a firewall question.
4. The browser is opened and the splash screen closed.
5. The server runs until the close button is pressed, or until no browser
   tab has been connected for a while and no pipeline is running.
"""

import argparse
import atexit
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

from . import APP_NAME, PANEL_EXTENSIONS, paths

logger = logging.getLogger("bacpipe_app")

HOST = "127.0.0.1"
DEFAULT_PORT = 5006
PING_TEXT = "bacpipe_app is running"
#: seconds without a connected browser tab after which the app closes
DEFAULT_IDLE_TIMEOUT = 300

# Loaded by every page. It asks the server every 1.5 seconds whether it is
# still there. When the application has been closed (close button in any
# stage, also the one of the dashboard), the page tries to close its tab.
# Browsers only allow a page to close a tab in some situations (Chrome and
# Edge do for a tab that was opened by the application and never navigated,
# Firefox does not), so otherwise the page is replaced by a short message
# instead of staying behind as a dead page.
PAGE_SCRIPT = """
(function () {
  var failures = 0;
  function closed() {
    window.close();
    setTimeout(function () {
      document.title = "bacpipe (closed)";
      document.body.innerHTML =
        '<div style="font-family: sans-serif; text-align: center; margin-top: 18vh; color: #333">' +
        '<h2>bacpipe has been closed</h2><p>You can close this tab.</p></div>';
    }, 400);
  }
  function check() {
    fetch("ping", { cache: "no-store" })
      .then(function (response) { failures = response.ok ? 0 : failures + 1; })
      .catch(function () { failures += 1; })
      .then(function () {
        if (failures >= 2) { closed(); } else { setTimeout(check, 1500); }
      });
  }
  setTimeout(check, 4000);
})();
"""


# --------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------
def setup_logging():
    """
    Log to ``app_state/app.log``.

    Started through the Windows launcher there is no console: stdout and
    stderr do not exist. They are pointed to the log file, so that ``print``
    and tracebacks of bacpipe and its dependencies end up somewhere useful.
    """
    log_file = paths.app_log_file()
    try:
        if log_file.exists() and log_file.stat().st_size > 2_000_000:
            log_file.unlink()
    except OSError:
        pass
    if sys.stdout is None or sys.stderr is None:
        stream = open(log_file, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    # the pages ask "/ping" every 1.5 seconds, do not log each request
    logging.getLogger("tornado.access").setLevel(logging.WARNING)


def url_for(port):
    return f"http://localhost:{port}/"


def running_instance_port():
    """Port of an application that is already running, or None."""
    try:
        port = json.loads(paths.instance_file().read_text(encoding="utf-8"))["port"]
        with urllib.request.urlopen(f"http://{HOST}:{port}/ping", timeout=2) as response:
            if response.read().decode() == PING_TEXT:
                return port
    except Exception:
        pass
    return None


def start_splash():
    """Start the splash screen process, returns None if that is not possible."""
    try:
        image = splash_image()
        return subprocess.Popen(
            paths.child_command("splash.py", str(image)),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=paths.child_env(),
            **paths.child_popen_kwargs(),
        )
    except Exception:
        logger.exception("could not show the splash screen")
        return None


def splash_image(width=380):
    """
    The logo with the "bacpipe" writing, scaled down for the splash screen.

    Tk can only load an image in its original size, so a small copy is
    created once and kept in the app state folder.
    """
    source = paths.logo_path()
    target = paths.state_dir() / "cache" / f"splash_{width}_{int(source.stat().st_mtime)}.png"
    if not target.exists():
        from PIL import Image

        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            height = round(image.height * width / image.width)
            image.convert("RGBA").resize((width, height), Image.LANCZOS).save(target)
    return target


def close_splash(splash):
    if splash is None:
        return
    try:
        splash.stdin.close()  # the splash closes itself when the pipe closes
        splash.wait(timeout=2)
    except Exception:
        try:
            splash.kill()
        except Exception:
            pass


# --------------------------------------------------------------------
# The server
# --------------------------------------------------------------------
def create_server(state, port):
    """
    Create the Panel server on the first free port starting at ``port``.

    Returns
    -------
    tuple
        (bokeh server, port)
    """
    import panel as pn
    from tornado.web import RequestHandler

    from .help_page import HelpHandler, ReadmeAssetHandler
    from .ui import SITE_TITLE, Page

    # The extensions have to be known before the first page is created,
    # including the ones only the dashboard of stage 3 needs.
    pn.extension(*PANEL_EXTENSIONS, js_files={"bacpipe_app": "bacpipe_app.js"})

    class PingHandler(RequestHandler):
        def get(self):
            self.set_header("Cache-Control", "no-store")
            self.write(PING_TEXT)

    class PageScriptHandler(RequestHandler):
        def get(self):
            self.set_header("Content-Type", "application/javascript; charset=utf-8")
            self.write(PAGE_SCRIPT)

    def make_page():
        return Page(state).template

    last_error = None
    for candidate in range(port, port + 50):
        try:
            server = pn.serve(
                {"/": make_page},
                port=candidate,
                address=HOST,
                websocket_origin=[f"localhost:{candidate}", f"{HOST}:{candidate}"],
                show=False,
                start=False,
                threaded=False,
                verbose=False,
                title=SITE_TITLE,
                extra_patterns=[
                    (r"/help", HelpHandler),
                    (r"/readme_assets/(.*)", ReadmeAssetHandler),
                    (r"/ping", PingHandler),
                    (r"/bacpipe_app.js", PageScriptHandler),
                ],
            )
            return server, candidate
        except OSError as error:  # port in use
            last_error = error
    raise RuntimeError(f"No free port found for the application: {last_error}")


def watch_for_idle(server, state, timeout):
    """
    Close the application when nobody is using it any more.

    Without this, closing the browser tab would leave an invisible process
    behind. The app stays alive while a pipeline is running, also without an
    open tab; starting the program again then reconnects to it.
    """
    from tornado.ioloop import PeriodicCallback

    last_active = time.monotonic()

    def check():
        nonlocal last_active
        if state.busy or len(server.get_sessions("/")) > 0:
            last_active = time.monotonic()
        elif time.monotonic() - last_active > timeout:
            logger.info("no browser tab connected for %s s, closing", timeout)
            state.shutdown()

    callback = PeriodicCallback(check, 5000)
    callback.start()
    return callback


def selftest(port):
    """
    Used by the tests of the build (``bacpipe --selftest``): request the
    main page and the help page from the running server.

    Requesting the main page makes the server create a complete page
    including the form, so this fails if the form can not be built.
    """
    for path, expected in (("", "bacpipe"), ("help", "bacpipe help"), ("ping", PING_TEXT)):
        with urllib.request.urlopen(f"http://{HOST}:{port}/{path}", timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            if response.status != 200 or expected not in body:
                raise RuntimeError(f"unexpected answer for /{path}")
        print(f"selftest: /{path} ok", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog=APP_NAME, description="bacpipe application")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="first port to try")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--no-splash", action="store_true", help="do not show the splash screen")
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=int(os.environ.get("BACPIPE_APP_IDLE_TIMEOUT", DEFAULT_IDLE_TIMEOUT)),
        help="close after this many seconds without a browser tab (0 = never)",
    )
    parser.add_argument("--selftest", action="store_true", help="start, check the pages, exit")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(argv)

    info = paths.build_info()
    if args.version:
        print(f"{APP_NAME} {info['version']} {info.get('commit', '')}".strip())
        return 0

    setup_logging()
    logger.info("starting %s %s", APP_NAME, info["version"])

    # 1. already running? then just show it
    if not args.selftest:
        port = running_instance_port()
        if port is not None:
            logger.info("already running on port %s", port)
            if not args.no_browser:
                webbrowser.open(url_for(port))
            return 0

    # 2. splash screen
    quiet = args.selftest or args.no_browser or args.no_splash
    splash = None if quiet else start_splash()

    exit_code = 0
    try:
        # 3. web server
        from .ui import AppState

        state = AppState()
        server, port = create_server(state, args.port)
        io_loop = server.io_loop

        def shutdown():
            """Stop the pipeline (if any) and the server. Safe from any thread."""
            state.stop_run()
            io_loop.add_callback(io_loop.stop)

        state.shutdown = shutdown
        atexit.register(state.stop_run)

        if args.selftest:

            def run_selftest():
                nonlocal exit_code
                try:
                    selftest(port)
                except Exception as error:
                    print(f"selftest failed: {error}", flush=True)
                    exit_code = 1
                shutdown()

            threading.Thread(target=run_selftest, daemon=True).start()
        else:
            paths.instance_file().write_text(
                json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8"
            )
            if args.idle_timeout > 0:
                watch_for_idle(server, state, args.idle_timeout)
            if info["bundled"]:
                threading.Thread(
                    target=lambda: setattr(
                        state, "update_notice", _update_notice(info["version"])
                    ),
                    daemon=True,
                ).start()

            def ready():
                # 4. the server is listening: show the page
                logger.info("serving on %s", url_for(port))
                close_splash(splash)
                if not args.no_browser:
                    webbrowser.open(url_for(port))

            io_loop.add_callback(ready)

        # 5. run until shutdown() is called
        server.start()
        io_loop.start()
    except Exception:
        logger.exception("the application could not be started")
        exit_code = 1
    finally:
        close_splash(splash)
        if not args.selftest:
            try:
                paths.instance_file().unlink()
            except OSError:
                pass
    logger.info("closed")
    return exit_code


def _update_notice(version):
    from .updates import update_notice

    return update_notice(version)
