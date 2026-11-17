# bacpipe application

A standalone build of bacpipe that runs without a Python installation. It
serves three stages in one browser page:

1. **Configure**: a form with all options of `config.yaml` and `settings.yaml`
2. **Run**: the live pipeline output, with a **Stop** button
3. **Explore**: the bacpipe dashboard

## Running from a checkout

```bash
python app/run_app.py
```

Options:

| Option | Meaning |
|---|---|
| `--port N` | first port to try (default 5006) |
| `--no-browser` | do not open the browser |
| `--no-splash` | do not show the splash screen |
| `--idle-timeout S` | close after `S` seconds without a connected tab (default 300, 0 = never) |
| `--selftest` | start, request the pages, exit |
| `--version` | print the version and exit |

Environment variables:

| Variable | Meaning |
|---|---|
| `BACPIPE_WORKSPACE` | workspace folder (default `~/bacpipe`) |
| `BACPIPE_APP_IDLE_TIMEOUT` | default for `--idle-timeout` |
| `BACPIPE_NO_UPDATE_CHECK=1` | disable the check for a newer release |

## Building

```bash
pip install -r app/requirements-build.txt
python app/build.py
```

A build only targets the system it runs on. The output in `app/dist/`:

```
bacpipe/                                  bacpipe.app/ on macOS
    bacpipe.exe | bacpipe                 launcher
    bacpipe_debug.cmd                     launcher with console (Windows)
    README.txt, LICENSE
    runtime/                              Python with bacpipe installed
bacpipe-<version>-<system>-<arch>.zip     .tar.gz on Linux and macOS
```

The build downloads a pinned
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
CPython, installs the requirements and bacpipe into it, copies `bacpipe_app`,
prunes development files, precompiles bytecode, creates the launcher, checks
path lengths and packs the archive.

| Option | Meaning |
|---|---|
| `--requirements X` | `pyproject`, or a requirements file in the repository root (default `requirements_no_cuda.txt`) |
| `--installer auto\|uv\|pip` | default: uv if installed, otherwise pip |
| `--link-mode hardlink` | uv only: link packages from the uv cache instead of copying them |
| `--no-compile`, `--no-archive` | skip bytecode compilation or the archive |
| `--output DIR` | output folder (default `app/dist`) |

Python version, checksums, extra packages, prune patterns and the Windows
path limit are set in [`build_config.toml`](build_config.toml).

## Tests

```bash
python -m pytest app/tests                                   # application code
python -m pytest app/tests --app-dist app/dist               # plus a built application
python -m pytest app/tests --app-dist app/dist \
       --app-e2e-model batdetect2_clip_avg                   # plus a pipeline run inside it
```

| File | Covers |
|---|---|
| `test_schema.py` | form generation from the YAML files and `form_overrides.yaml` |
| `test_contract.py` | the bacpipe names the application depends on (`BACPIPE_CONTRACT`, `PANEL_EXTENSIONS`) |
| `test_app.py` | widgets, remembered values, process handling, help page, `--selftest` |
| `test_build.py` | a built application; skipped without `--app-dist` |

`python -m bacpipe_app.e2e_check` runs the pipeline and builds the dashboard
in the terminal, without a browser.

## Releases

[`.github/workflows/build_app.yaml`](../.github/workflows/build_app.yaml)
builds and tests on Windows, Linux and macOS on every push to `main`, weekly,
on demand, and for pull requests touching the application or its
dependencies. When a release is published, the archives are attached to it.

To release, set the version in `pyproject.toml` and publish a GitHub release
with the tag `v<version>`. The version shown in the application, the archive
name and the update notice all derive from it.

## The form

The form is generated from `bacpipe/config.yaml` and `bacpipe/settings.yaml`
at startup:

| YAML | Form |
|---|---|
| key | field |
| comment above a key | help text |
| `#####  TITLE  #####` banner | collapsible section |
| `True` / `False` | radio buttons |
| number / text | input |
| list | multi-select dropdown |
| nested mapping | box with one field per entry |

New settings appear automatically. Dropdown options, folder pickers, hidden
fields, labels, descriptions and the sections that start unfolded are set in
[`bacpipe_app/form_overrides.yaml`](bacpipe_app/form_overrides.yaml); the
available hints are listed at the top of that file. Every top-level key needs
a banner above it, otherwise it lands in a section called "Other".

Colors and font sizes are set in [`bacpipe_app/theme.yaml`](bacpipe_app/theme.yaml).

## Files written

Everything is written to the workspace (`~/bacpipe`), never to the
application folder:

| Path | Content |
|---|---|
| `bacpipe_results/` | results (configurable in the form) |
| `bacpipe_model_checkpoints/` | model checkpoints (configurable in the form) |
| `app_state/last_values.yaml` | last form values |
| `app_state/app.log` | application log |
| `app_state/runs/<time>/` | `run.yaml` and `terminal_output.log` of each run |

Relative paths in the settings are resolved against the workspace.

## Code layout

```
app/
├── run_app.py              run from a checkout
├── build.py                build script
├── build_config.toml       build configuration
├── bacpipe_app/
│   ├── server.py           startup, web server, shutdown
│   ├── ui.py               pages of the three stages
│   ├── schema.py           YAML files -> field descriptions
│   ├── form.py             field descriptions -> Panel widgets
│   ├── form_overrides.yaml form hints
│   ├── runner.py           starts and watches the pipeline process
│   ├── worker.py           pipeline process
│   ├── dashboard.py        builds the dashboard for stage 3
│   ├── help_page.py        README as help page
│   ├── splash.py           splash screen process
│   ├── dialogs.py          native file and folder dialogs process
│   ├── updates.py          update notice
│   ├── e2e_check.py        pipeline + dashboard without a browser
│   └── paths.py            file locations
├── launcher/               Windows and Unix launchers, README.txt for users
└── tests/
```

## Troubleshooting

| Symptom | Where to look |
|---|---|
| Application does not start | run `bacpipe_debug.cmd` (Windows) or the launcher in a terminal; `~/bacpipe/app_state/app.log` |
| Browser does not open | the address is logged in `app.log` ("serving on ...") |
| Pipeline fails | the run view; `app_state/runs/<time>/terminal_output.log` |
| A setting is missing in the form | it is `hidden` in `form_overrides.yaml` |
| Build: checksum mismatch | `[python.sha256]` in `build_config.toml` |
| Build: path too long | add the folder to `[prune]` in `build_config.toml` |
| `test_contract.py` fails | bacpipe changed a name the application uses; the message says what to adapt |
