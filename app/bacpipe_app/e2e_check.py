"""
Run the pipeline on the bundled example data and build the dashboard the same
way the application does, without a browser.

    python -m bacpipe_app.e2e_check --model batdetect2_clip_avg

Exits with 0 if both succeed.
"""

import argparse
import sys
import time

from . import paths, schema
from .runner import PipelineRun, absolute_path_values
from .ui import validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="batdetect2_clip_avg")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds")
    parser.add_argument("--no-dashboard", action="store_true", help="only run the pipeline")
    args = parser.parse_args(argv)

    definitions = schema.load_form_definitions(paths.bacpipe_package_dir())
    values = {}
    for name, specs in definitions.items():
        folder_keys = [s.key for s in specs if s.kind == "folder"]
        values[name] = absolute_path_values(schema.defaults(specs), folder_keys)
    config, settings = values["config"], values["settings"]
    config["models"] = [args.model]
    config["audio_dir"] = paths.TEST_DATA_AUDIO_DIR

    problems = validate(config, settings)
    if problems:
        print("\n".join(problems))
        return 1

    run = PipelineRun(config, settings)
    run.start()
    cursor, started = 0, time.time()
    while run.running:
        text, cursor = run.output.read_from(cursor)
        sys.stdout.write(text)
        sys.stdout.flush()
        if time.time() - started > args.timeout:
            run.stop()
            print(f"\ne2e_check: timeout after {args.timeout} s")
            return 1
        time.sleep(0.2)
    text, cursor = run.output.read_from(cursor)
    sys.stdout.write(text)

    if not run.succeeded:
        print(f"\ne2e_check: the pipeline failed (exit code {run.returncode})")
        return 1
    if not run.wants_dashboard:
        print("\ne2e_check: the pipeline did not hand over to the dashboard")
        return 1
    print("\ne2e_check: pipeline ok", flush=True)
    if args.no_dashboard:
        return 0

    from .dashboard import build_dashboard

    layout = build_dashboard(config, settings, run.dashboard_kwargs_file)
    print(f"e2e_check: dashboard ok ({type(layout).__name__} with {len(layout)} tabs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
