# The bacpipe application

This folder turns bacpipe into a program that people can download, unpack and
start with a double-click, without installing Python.

Starting the program gives the same result as `python run_pipeline.py`, in
three stages inside one browser page:

| Stage | What the user sees | Replaces |
|---|---|---|
| 1. Configure | A form with every option of `config.yaml` and `settings.yaml`, and a big **Run pipeline** button | editing the two YAML files |
| 2. Run | The terminal output of the pipeline, including progress bars, with a **Stop** button | the terminal |
| 3. Explore | The bacpipe dashboard | the dashboard that `run_pipeline.py` opens |

A **Help** button in the header opens the README of the repository. The
sidebar with logo, contact text and close button is the same in all stages.

Colors and font sizes that are a matter of taste (the terminal, the help
texts, the play button) are in
[`bacpipe_app/theme.yaml`](bacpipe_app/theme.yaml): change a hex code, save,
restart.

Contents

1. [Quick start for developers](#1-quick-start-for-developers)
2. [What is verified and what is not](#2-what-is-verified-and-what-is-not)
3. [How it works](#3-how-it-works)
4. [The form](#4-the-form)
5. [The build](#5-the-build)
6. [Tests](#6-tests)
7. [Automatic builds and releases](#7-automatic-builds-and-releases)
8. [Keeping the application in step with `main`](#8-keeping-the-application-in-step-with-main)
9. [Restricted computers and unsigned programs](#9-restricted-computers-and-unsigned-programs)
10. [Why it is built this way](#10-why-it-is-built-this-way)
11. [Maintenance recipes](#11-maintenance-recipes)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Quick start for developers

**Run the application from the checkout** (no build needed, uses your
existing bacpipe environment):

```bash
python app/run_app.py
```

**Build the program** for the system you are on:

```bash
pip install -r app/requirements-build.txt
python app/build.py
```

The result is in `app/dist/`:

```
app/dist/bacpipe/                          the folder that is shipped
    bacpipe.exe                            start this          (Linux: bacpipe)
    bacpipe_debug.cmd                      same, with a console (Windows only)
    README.txt                             instructions for users
    LICENSE
    runtime/                               Python with bacpipe installed
app/dist/bacpipe-1.3.5-windows-x86_64.zip  the folder, packed for download
```

On macOS the folder is `bacpipe.app` and the archive a `.tar.gz`.

A program can only be built on the system it is for: the Windows program on
Windows, and so on. GitHub Actions does this for all three systems on every
push to `main` (see [section 7](#7-automatic-builds-and-releases)), so you do
not need a Windows machine to produce the Windows program.

**Test:**

```bash
python -m pytest app/tests                        # the code, about 3 seconds
python -m pytest app/tests --app-dist app/dist    # plus the built program
```

---

## 2. What is verified and what is not

Read this before relying on a build.

Verified on Linux, on the machine the application was developed on:

- All three stages in a real browser (headless Chrome): the form, a pipeline
  run with live output and progress bars, the switch to the dashboard, and
  the error screen when a run fails.
- A complete build (4.1 GB folder), moved to another location and started
  with a deliberately broken `PYTHONPATH`: starts and passes all tests in
  `app/tests/test_build.py`, including a pipeline run and the dashboard
  inside the built program.
- The Windows launcher `bacpipe.exe` compiles (cross-compiled from Linux) and
  contains icon, version information and the manifest. It was inspected, not
  executed.

**Not verified: the Windows and macOS builds have never been run.** There was
no Windows or macOS machine available. The build script and the tests are
written for all three systems, and the workflow
`.github/workflows/build_app.yaml` builds and tests on real Windows and macOS
machines, so **the first run of that workflow is the actual test** of those
two. Expect that a detail needs fixing there. The points most likely to need
attention:

| Point | Where it shows up |
|---|---|
| Does `bacpipe.exe` start `pythonw.exe` correctly | test `test_launcher_starts_the_application` |
| Is `msvcp140.dll` shipped by a package ([details](#the-c-runtime-on-windows)) | test `test_cpp_runtime_libraries_are_shipped` |
| Longest path on Windows | build step 7, test `test_paths_are_short_enough_for_windows` |
| Splash screen and folder dialog on Windows/macOS | only by hand, they need a screen |
| Size of the Windows `.zip` (GitHub releases allow 2 GB per file) | end of the build log |
| Does `bacpipe.app` open by double-click on macOS | only by hand |

Also not tested anywhere: a GPU build, and a machine without the Microsoft
C++ runtime installed (GitHub's Windows machines have it). See
[section 9](#9-restricted-computers-and-unsigned-programs) for how to test on
a clean Windows.

---

## 3. How it works

### No freezing

Tools like PyInstaller analyse the code to find out which modules it needs
and copy those into a program. That analysis is where such builds break: it
misses modules that are imported dynamically (bacpipe imports its model
pipelines by name), and every large dependency (torch, tensorflow, jax,
transformers, numba) needs special handling that changes with its versions.

This build does not analyse anything. It takes a complete, relocatable Python,
runs `pip install` into it exactly like a user would, and adds a small
launcher:

```
bacpipe.exe  ──starts──▶  runtime\pythonw.exe -m bacpipe_app
```

The consequence is the property that matters most for maintenance: **if
`pip install` gives a working bacpipe, the program works.** A new model, a new
dependency or a refactoring in bacpipe needs no change here.

The price is size (the folder contains complete packages, about 4 GB with the
CPU versions of torch and tensorflow). A frozen build would not be much
smaller, because torch and tensorflow make up most of it either way.

### The processes

```
bacpipe.exe            launcher (small C program), waits and passes on the exit code
 └─ pythonw -m bacpipe_app        the application: a Panel web server on
     │                            127.0.0.1, shows the three stages
     ├─ python splash.py          splash screen while starting
     ├─ python dialogs.py         "Browse" folder dialog, when clicked
     └─ python worker.py          the pipeline = `bacpipe.play()`, one per run
```

- The **application process** starts in about 1.5 seconds, because it does
  not import bacpipe (which loads torch and tensorflow, 7 to 8 seconds). It
  reads `config.yaml`, `settings.yaml` and `core/constants.py` directly from
  the installed package.
- The **pipeline runs in its own process** (`worker.py`). That is what makes
  the terminal view exact (everything the process prints is shown, including
  messages from tensorflow's C++ code), lets the user stop a run, and frees
  the memory of the models afterwards.
- For **stage 3** the application process imports bacpipe and builds the
  dashboard inside its own page.

### How the pipeline hands over to the dashboard

`bacpipe.play()` normally ends by calling `visualize_using_dashboard(**kwargs)`,
which starts a web server of its own. The application already is a web
server, so:

1. `worker.py` replaces `visualize_using_dashboard` with a function that
   saves its arguments to a file and returns.
2. When the worker has ended, `dashboard.py` reads that file and creates
   `DashBoard(models, **kwargs)` - the same object the function would have
   created - and puts its layout into the page.

This is the only place where the application depends on bacpipe internals.
The names it uses are listed in `BACPIPE_CONTRACT` (`bacpipe_app/dashboard.py`)
and checked by `app/tests/test_contract.py`, which reads the bacpipe source
files and fails with an instruction if one of them changed.

### Where the program writes

Never into its own folder (a test checks this), always into the workspace
`~/bacpipe` (`C:\Users\<name>\bacpipe`):

| Path | Content |
|---|---|
| `bacpipe_results/` | embeddings and evaluations (changeable in the form) |
| `bacpipe_model_checkpoints/` | downloaded checkpoints (changeable in the form) |
| `app_state/last_values.yaml` | what was entered in the form last time |
| `app_state/app.log` | log of the application |
| `app_state/runs/<time>/` | per run: `run.yaml` (the values used) and `terminal_output.log` |

Relative paths in the settings are resolved against the workspace, the same
way they are resolved against the repository folder for `run_pipeline.py`.
The environment variable `BACPIPE_WORKSPACE` moves the workspace.

### Behaviour worth knowing

- **Starting it twice** opens the page of the running program again instead
  of starting a second one.
- **Closing the browser tab** does not stop a running pipeline. Start the
  program again to get back to it. Without a pipeline running, the program
  ends itself five minutes after the last tab was closed
  (`--idle-timeout`, `BACPIPE_APP_IDLE_TIMEOUT`).
- **The form remembers** the last values; "Reset to defaults" restores the
  YAML defaults.
- **Update notice**: a built program asks the GitHub API once per start
  whether a newer release exists and shows a link in the sidebar if so. No
  user data is sent; `BACPIPE_NO_UPDATE_CHECK=1` switches it off. It only
  finds something once releases with a version tag exist (section 7).
- The web server only listens on `127.0.0.1`. It is not reachable from other
  computers.
- **Closing the program** (any close button, including the one of the
  dashboard): every open page notices within a few seconds that the program
  is gone and tries to close its tab. Browsers only allow that in some
  situations (Chrome and Edge for a tab that the program opened and that was
  never reloaded, Firefox not at all), so otherwise the page is replaced by
  "bacpipe has been closed. You can close this tab."
- **File and folder dialogs** are the ones of the system: on Windows and
  macOS through tkinter, on Linux through `zenity` or `kdialog`
  (`dialogs.py`). Only a Linux without both falls back to the dated dialog
  that tkinter draws itself.

### The files

```
app/
├── README.md                 this file
├── run_app.py                run the application from the checkout
├── build.py                  the build script
├── build_config.toml         Python version, package list, prune list
├── requirements-build.txt    what build.py and the tests need
├── bacpipe_app/              the application (copied into the runtime)
│   ├── __main__.py           `python -m bacpipe_app`
│   ├── server.py             start: splash, web server, browser, shutdown
│   ├── ui.py                 the pages of the three stages
│   ├── schema.py             YAML file  ->  description of the form fields
│   ├── form_overrides.yaml   hints for fields (dropdown options, ...)
│   ├── form.py               description of the fields  ->  Panel widgets
│   ├── theme.yaml            colors and font sizes, meant to be edited
│   ├── theme.py              reads theme.yaml
│   ├── runner.py             starts and watches the pipeline process
│   ├── worker.py             the pipeline process
│   ├── dashboard.py          builds the bacpipe dashboard for stage 3
│   ├── help_page.py          README as web page
│   ├── readme_links.py       link handling of the README
│   ├── splash.py             splash screen (own process)
│   ├── dialogs.py            file and folder dialogs (own process)
│   ├── updates.py            "newer version available"
│   ├── e2e_check.py          pipeline + dashboard without a browser
│   └── paths.py              where everything is
├── launcher/
│   ├── windows/              launcher.c, icon/version/manifest resources,
│   │                         bacpipe_debug.cmd
│   ├── unix/bacpipe.sh.in    launcher for Linux and macOS
│   └── README_<system>.txt   the README.txt that users get
└── tests/
```

---

## 4. The form

The form is **generated** from `bacpipe/config.yaml` and
`bacpipe/settings.yaml` each time the program starts. Nothing about the
individual settings is written down in the application code.

| In the YAML file | In the form |
|---|---|
| a key | a field, labelled with the key |
| the comment lines above the key | the help text below the field |
| a banner `#####  TITLE  #####` | a collapsible section |
| a banner, an empty line, another banner | a heading above the sections that follow ("Evaluation Settings") |
| value `True` / `False` | radio buttons |
| a number | number input |
| a text | text input |
| a list | dropdown with multiple selection |
| a nested mapping (`probe_configs`) | a box with one field per entry |

**A new setting in a YAML file appears in the form automatically**, with the
widget for its type and its comment as help text. A test proves this with an
invented key (`test_a_new_setting_needs_no_code_change`).

What cannot be read from the YAML file is declared in
[`bacpipe_app/form_overrides.yaml`](bacpipe_app/form_overrides.yaml), a short
file with one entry per field that needs a hint:

```yaml
config:
  models:
    options_from: supported_models        # bacpipe.core.constants
  dim_reduction_model:
    options_from: dim_reduction_models    # files in dimensionality_reduction/
settings:
  device:
    options: [cpu, cuda, mps]             # -> dropdown
  nr_parallel_workers:
    widget: yaml                          # `False` or a number
  main_results_dir:
    widget: folder                        # text input + Browse button
  dashboard_port:
    hidden: true                          # the app chooses the port itself
sections:
  settings:
    expanded: [Path Settings, ...]        # sections that start unfolded
```

All available hints are explained at the top of that file. The model list
comes from `bacpipe.core.constants.supported_models`, so a new model appears
in the dropdown without any change here.

Two things to know when writing YAML comments, since they are now shown to
users:

- A comment directly above a key is that key's help text. A comment that is
  about something else (like the note for Windows users about quotation
  marks above `audio_dir`) ends up in the help text too. Use a `description:`
  hint to replace such a text, as done for `audio_dir`.
- A top level key needs a banner somewhere above it, otherwise it lands in a
  section called "Other". A test reports keys without a section.

One field has behaviour of its own, written in `ui.py`
(`_link_annotations_to_audio_dir`): `annotations_filename`. bacpipe expects
the name of a file inside the audio folder. The form shows the complete path
with a file dialog instead, moves it along when the audio folder changes
(unless a file somewhere else was chosen), and hands bacpipe the name
relative to the audio folder, or the complete path if the file is elsewhere.
If the file does not exist while `only_embed_annotations` is True, a note
appears below the field; it does not block a run.

The section titles are the banners of the YAML file in title case. The
section "Using Your Anootation Settings" is spelled that way because the
banner in `settings.yaml` is; correcting the banner needs the same correction
in the `expanded` list (a test points that out).

Limits of the form: entries cannot be *added* to `probe_configs` and
`clust_configs` (the existing ones can be edited and switched on and off),
and a list field only offers the values that are in the YAML default or in an
`options` hint.

---

## 5. The build

`python app/build.py` runs eight steps. Each is one function in
[`build.py`](build.py), in this order:

| Step | Function | What it does |
|---|---|---|
| 1 | `download_python` | Downloads the pinned standalone Python (once, kept in `app/.cache/`) and checks its checksum |
| 2 | `install_packages` | `pip install` of the dependencies and of bacpipe into that Python |
| 3 | `install_app` | Copies `bacpipe_app`, the README with its images, writes `build_info.json` |
| 4 | `prune` | Deletes C/C++ development files that are not needed to run anything |
| 5 | `compile_bytecode` | Precompiles `.pyc` files: faster first start, no writes into the folder later |
| 6 | `make_launcher` | Windows: compiles `bacpipe.exe`; Linux/macOS: writes the start script; macOS: `Info.plist` and icon |
| 7 | `check_paths` | Stops the build if a path is too long for Windows |
| 8 | `make_archive` | `.zip` on Windows, `.tar.gz` elsewhere |

Options (`python app/build.py --help`):

| Option | Meaning |
|---|---|
| `--requirements X` | `pyproject`, or a requirements file in the repository root |
| `--installer uv\|pip` | default: uv if installed (much faster), otherwise pip |
| `--link-mode hardlink` | uv only: do not copy packages out of the uv cache. Saves several GB for local test builds |
| `--no-archive`, `--no-compile` | skip a step for quick local builds |
| `--output DIR` | default `app/dist` |

Everything that is a choice is in
[`build_config.toml`](build_config.toml), with comments.

### Which packages are installed

By default the packages of `requirements_no_cuda.txt` on all three systems,
because that is the set the existing test workflows install and test. The
build installs them and then bacpipe itself with `--no-deps`.

`--requirements pyproject` installs the dependencies declared in
`pyproject.toml` instead. On Linux that includes the CUDA packages and gives a
GPU program of several GB more.

A GPU program for Windows needs a requirements file that installs the CUDA
builds of torch (from `https://download.pytorch.org/whl/cu...`); create one
next to `requirements_no_cuda.txt` and pass it with `--requirements`.

Note that a build uses the requirements file **as it is in your working
directory**. If you have uncommitted changes in it, your local build differs
from the build on GitHub.

### The Python inside the program

A build of CPython from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone)
(the builds `uv` uses): complete, relocatable, the same for all three
systems, including tkinter for the splash screen and the folder dialog. The
version and the checksums are pinned in `build_config.toml`.

### The Windows launcher

[`launcher/windows/launcher.c`](launcher/windows/launcher.c), a short C
file: find the own folder, start `runtime\pythonw.exe -E -s -X utf8 -m
bacpipe_app`, wait, return its exit code. It is compiled with the zig
compiler from the pip package `ziglang`, so the build needs no Visual Studio.
`launcher.rc.in` adds the icon and the version information,
`launcher.manifest` states that the program runs with the rights of the user.

`bacpipe_debug.cmd` does the same with `python.exe`, which shows a console
with all messages.

### The C++ runtime on Windows

torch and tensorflow need Microsoft's C++ runtime libraries
(`msvcp140.dll`, `vcruntime140.dll`). Most Windows computers have them, a
freshly installed one does not, and installing them needs administrator
rights. So they have to be inside the folder. `vcruntime140.dll` comes with
the Python runtime and torch ships its own copies in `torch\lib`. Whether
that is enough for tensorflow could not be checked without a Windows machine.

The test `test_cpp_runtime_libraries_are_shipped` checks that `msvcp140.dll`
is somewhere in the folder. If it fails, or if tensorflow fails to load on a
clean Windows, add a package that ships the libraries to
`[extra_requirements] windows` in `build_config.toml`. The pip package
`msvc-runtime` does that, but its current version has no build for Python
3.11, so pin a version that has one.

---

## 6. Tests

```bash
python -m pytest app/tests
```

| File | What it protects | Needs |
|---|---|---|
| `test_schema.py` | Every YAML key becomes a field; the untouched form equals the YAML file; hints in `form_overrides.yaml` refer to existing keys; the model list is `supported_models` | pyyaml |
| `test_contract.py` | The names the application uses from bacpipe still exist. Reads the source, does not import bacpipe | nothing |
| `test_app.py` | Widgets and their values, remembered values, the process handling, the help page, and a start of the whole application (`--selftest`) | panel |
| `test_build.py` | A built program, see below. Skipped without `--app-dist` | a build |

The first three take about three seconds together and need neither torch nor
tensorflow. They are worth running in every push (they do, see section 7).

Tests of a built program:

```bash
python app/build.py
python -m pytest app/tests/test_build.py --app-dist app/dist \
       --app-e2e-model batdetect2_clip_avg
```

- the folder is complete, the version equals `pyproject.toml`, the Python is
  the pinned one
- the runtime ignores `PYTHONPATH`/`PYTHONHOME` of the machine and loads
  everything from its own folder
- `import bacpipe, torch, tensorflow, ...` works inside the runtime. This is
  the test that finds a dependency missing from the requirements file, or a
  prune pattern that deletes too much
- **the real launcher starts the application** (`bacpipe.exe --selftest`):
  the web server starts, the form page and the help page are requested, the
  exit code is passed on
- the program does not write into its own folder
- Windows only: no console window, no request for administrator rights, path
  lengths, C++ runtime
- with `--app-e2e-model`: the pipeline runs on the example data and the
  dashboard is built, inside the built program. `batdetect2_clip_avg` is used
  because it needs no checkpoint download, which keeps the test independent
  of the network

Two commands that are useful beyond the tests:

```bash
bacpipe --selftest                     # any build: starts, checks, exits
python -m bacpipe_app.e2e_check        # pipeline + dashboard in the terminal
```

What the tests do not cover: anything that needs a screen (splash screen,
folder dialog, the look of the pages) and the browser side of the terminal
view. Check those by hand after changes to `ui.py`, `splash.py` or
`dialogs.py`.

The existing bacpipe tests are untouched, `python -m pytest bacpipe/tests`
does not run the application tests and vice versa.

---

## 7. Automatic builds and releases

[`.github/workflows/build_app.yaml`](../.github/workflows/build_app.yaml)

| Trigger | What happens |
|---|---|
| push to `main` | application tests, then build and test on Windows, Linux and macOS. The archives are kept for 14 days as artifacts of the run (Actions tab → the run → Artifacts) |
| a release is published | the same, and the three archives are attached to the release |
| pull request | only if it touches `app/`, the YAML files, `bacpipe/core`, the dashboard code, `pyproject.toml` or a requirements file |
| every Monday | the same build, to notice breakage caused by new versions of dependencies |
| "Run workflow" button | on demand, for any branch |

### Making a release

1. Set the new version in `pyproject.toml` (the only place the version is
   written down; the program, the file name of the archive and the properties
   of `bacpipe.exe` take it from there).
2. On GitHub: Releases → "Draft a new release", create the tag `v<version>`,
   publish.
3. The workflow builds the three programs and attaches them to the release
   when it has finished (watch it in the Actions tab).

The update notice in the program compares its own version with the tag of
the latest release, so tags have to look like `v1.3.6`.

GitHub allows 2 GB per release file. If the Windows archive grows beyond
that, extend the prune list, or upload it somewhere else (Zenodo and Hugging
Face both accept larger files) and link it from the release.

---

## 8. Keeping the application in step with `main`

The design goal was that a change in bacpipe needs no manual work here. This
is what carries that, and what remains.

**What needs no action**

| Change in bacpipe | Why the program follows |
|---|---|
| new or renamed setting | the form is generated from the YAML files |
| new model | the dropdown reads `supported_models` |
| new dimensionality reduction | the dropdown lists the files of that folder |
| new or updated dependency | the build runs `pip install` |
| any change in pipeline or dashboard code | the program runs that code unchanged |
| README change | the help page is the README |
| new version number | read from `pyproject.toml` |

**What the tests catch**, in the normal run on every push, with a message
saying what to adapt:

| Change in bacpipe | Test |
|---|---|
| `play`, `visualize_using_dashboard`, `DashBoard`, `build_layout` renamed or called differently | `test_contract.py` |
| the dashboard uses a new Panel extension | `test_contract.py` → add it to `PANEL_EXTENSIONS` |
| a setting with a hint in `form_overrides.yaml` renamed | `test_schema.py` |
| a dependency missing from the requirements file | `test_build.py` (import test) |
| a new dependency with very long file paths | build step 7 |

**What remains manual**

- A new setting that should be a dropdown or a folder field needs a line in
  `form_overrides.yaml`. Without it the field is still there, as a text or
  number input.
- The contact text in the sidebar exists twice: in the bacpipe dashboard and
  in `bacpipe_app/ui.py` (`CONTACT_TEXT`). The link to the live demo is only
  in the sidebar of stages 1 and 2; the sidebar of stage 3 belongs to the
  dashboard code of bacpipe.
- The pinned Python (`build_config.toml`) should be updated a few times a
  year for security fixes, see section 11.

**Practices that keep it that way**

1. *Build on every push to `main`, not only for releases.* A build that
   breaks is then traced to one commit. This is what the workflow does.
2. *Build on a schedule as well.* Dependencies without an upper version bound
   can break a build without any commit; the weekly run shows that early.
3. *One source for the version* (`pyproject.toml`), and releases only through
   the workflow, never from a developer's machine. Then the program always
   corresponds to a commit, which `build_info.json` records.
4. *Keep the application thin.* Whenever something about a setting or a model
   can be read from bacpipe, read it instead of repeating it. Prefer a hint in
   `form_overrides.yaml` over code.
5. *Keep the contract small and tested.* If the application needs something
   new from bacpipe, add the name to `BACPIPE_CONTRACT`.
6. *Run the fast application tests together with the bacpipe tests*, so a
   contributor who renames `DashBoard` sees the failure in their pull
   request. The workflow does this for pull requests that touch the relevant
   paths.

One optional simplification inside bacpipe would remove the only piece of
indirection: splitting `visualize_using_dashboard` into a function that builds
the dashboard and one that serves it. `dashboard.py` could then call the
first directly instead of rebuilding its first half. It was not done here to
leave the bacpipe code untouched.

---

## 9. Restricted computers and unsigned programs

Many users are on computers managed by an institution, without administrator
rights. The program is built so that nothing it does asks for them.

### What the program avoids

| Would need admin rights or trigger a prompt | How it is avoided |
|---|---|
| An installer (always asks for elevation) | There is none. The program is a folder: unpack and start. Removing it is deleting the folder |
| Writing to `C:\Program Files`, the registry, or services | Never done. All data goes to the home folder |
| Windows Firewall prompt ("allow access"), which needs an admin to confirm | The server listens on `127.0.0.1` only. The prompt appears for programs that accept connections from other computers |
| Installing the Microsoft C++ runtime | Shipped inside the folder ([check](#the-c-runtime-on-windows)) |
| Enabling long paths in Windows | Paths inside the folder are kept below 200 characters, checked in every build |
| Windows guessing that a program is an installer and asking for elevation | The manifest of `bacpipe.exe` states `asInvoker` |
| Python of the user interfering, or the program changing it | The runtime ignores `PYTHONPATH` and user packages, and changes nothing outside its folder |

### What cannot be avoided without signing

`bacpipe.exe` is not signed. Two things follow.

**SmartScreen.** When a downloaded, unsigned program is started for the first
time, Windows shows "Windows protected your PC". The user clicks "More info"
→ "Run anyway". This needs no administrator rights, unless the institution
has configured SmartScreen to block instead of warn.

The message can be avoided: Windows marks downloaded files, and the Explorer
passes that mark on to everything it unpacks from a marked `.zip`. Unblocking
the `.zip` *before* unpacking (right-click → Properties → Unblock → OK)
removes the mark, and the program then starts without the message. This is
described in the `README.txt` users get. Unpacking with 7-Zip has the same
effect.

**Application control (AppLocker, WDAC).** Some institutions only allow
programs from an approved list. An unknown program is then blocked no matter
what, and only the IT department can change that. `bacpipe_debug.cmd` is
offered as an alternative start, because some configurations treat scripts
differently from `.exe` files, but that is not something to count on. For
these users the route is `pip install bacpipe`, if they have Python.

Antivirus programs sometimes flag small unsigned programs that start other
programs. The launcher carries complete version information and does nothing
but start Python, which keeps that risk low, but it cannot be excluded. If it
happens, report the file as a false positive to the vendor (Microsoft:
<https://www.microsoft.com/wdsi/filesubmission>).

### Signing, if the warnings turn out to be a problem

Signing `bacpipe.exe` (and ideally the DLLs) removes the "unknown publisher"
wording and lets IT departments approve the program by publisher. Options to
look into, with terms that change, so check the current conditions:

- **SignPath Foundation** offers free code signing for open source projects,
  integrated with GitHub Actions. The most fitting option for bacpipe.
- **Azure Trusted Signing** is a paid service by Microsoft with a low monthly
  price; eligibility is restricted to certain countries and organisation
  types.
- A classic code signing certificate from a certificate authority costs a
  few hundred euros per year and nowadays has to live on a hardware token or
  in a cloud HSM.

Signing does not make the SmartScreen message disappear on the first day:
SmartScreen builds up reputation per publisher with the number of downloads.

On macOS the equivalent is an Apple Developer account (99 USD per year) for
signing and notarisation. Without it, users open the app once with
right-click → Open, or remove the download mark with
`xattr -dr com.apple.quarantine bacpipe.app`, both described in the
`README.txt`.

### Testing like a restricted user

GitHub's Windows machines are administrator accounts with developer tools
installed, so they do not show these problems. Before announcing a release,
try the downloaded `.zip` once in **Windows Sandbox** (a free feature of
Windows 10/11 Pro: a clean, throw-away Windows) or a fresh virtual machine,
as a standard user:

1. download the `.zip` with a browser (so that it carries the download mark)
2. unpack it with the Explorer into `Downloads`
3. start `bacpipe.exe`, run the example data with one model

That one check covers SmartScreen, the C++ runtime, path lengths and the
firewall.

---

## 10. Why it is built this way

The common ways to ship a Python program, and how they fit bacpipe:

| Approach | Fit |
|---|---|
| **PyInstaller / Nuitka** (freeze the code into a program) | The standard for small tools. With torch, tensorflow, jax, transformers and dynamically imported model pipelines, the build needs a growing list of special cases that break with dependency updates. That is the opposite of "build `main` without manual fixing". Single-file mode also unpacks itself on every start, which is slow at this size and triggers antivirus programs |
| **A Python environment plus launcher** (this build) | Nothing to analyse, so nothing to fix when bacpipe changes. Large, and the code is readable in the folder, which is no drawback for open source |
| **Briefcase** (BeeWare) | Uses the same principle and adds installers. It expects the project in its own layout and is less transparent than one script you can read top to bottom. Worth a look if installers (`.msi`, `.dmg`) are wanted later |
| **conda constructor** | Produces installers from conda packages. bacpipe is installed with pip, and installers are what restricted users cannot run |
| **An installer** (`.msi`, Inno Setup) around the folder | Nicer for home users (start menu entry), but installers ask for elevation by default. Can be added on top of this build without changing it; an Inno Setup installer can be configured to install per user without admin rights |

General practices that this build follows, as a checklist for future changes:

- **Pin what goes in.** Python version with checksum, compiler version; and
  record what was built (`build_info.json`: version, commit, date).
- **Build in CI on the target system**, from a clean checkout, never by hand.
- **Test the result, not only the code**: start the real launcher on the real
  system and run a real pipeline.
- **One folder, no installation, no writes into the folder.**
- **Separate program and user data**, so that updating is replacing the
  folder and nothing is lost.
- **Tell users about new versions** (the update notice) rather than building
  an auto-updater, which needs signing to be trustworthy.
- **Show the output.** A program without a console must put its errors
  somewhere a user can find and send: the terminal view, the saved
  `terminal_output.log` and `app.log`.

---

## 11. Maintenance recipes

**Update the Python inside the program.** On
<https://github.com/astral-sh/python-build-standalone/releases> pick a
release, and in `build_config.toml` set `version`, `release` and the five
`sha256` values of the files named
`cpython-<version>+<release>-<platform>-install_only_stripped.tar.gz`. Stay
within `requires-python` of `pyproject.toml`.

**Give a field a dropdown, a folder button or hide it.** Add an entry in
`bacpipe_app/form_overrides.yaml`.

**Change colors or font sizes.** `bacpipe_app/theme.yaml`. Edit the YAML
file, not the `DEFAULTS` in `theme.py`: those only apply to values missing
from the YAML file.

**Change which sections start unfolded.** `sections:` in
`bacpipe_app/form_overrides.yaml`.

**Change the look of a page.** `bacpipe_app/ui.py`: `_sidebar`,
`_config_view`, `_run_view`, `_dashboard_view`. Try it with
`python app/run_app.py`.

**Add a file type or widget kind to the form.** Add the kind to `KINDS` in
`schema.py` and a class to `FIELD_CLASSES` in `form.py`.

**Use a sharper icon.** Point `icon` in `build_config.toml` to a square image
of at least 256×256. The current one is 64×64.

**Make the download smaller.** Add patterns to `[prune]` in
`build_config.toml`. The import test and the pipeline test tell you if a
pattern removed something that is needed.

**Add packages to the built program.** `[extra_requirements]` in
`build_config.toml`. Users can also do it themselves:
`runtime\python.exe -m pip install <package>`.

**Change what the launcher does.** `launcher/windows/launcher.c` and
`launcher/unix/bacpipe.sh.in`. If you change the options passed to Python,
change them in both, and in `bacpipe_debug.cmd`.

---

## 12. Troubleshooting

| Symptom | Where to look |
|---|---|
| Program does not start, nothing visible | Start `bacpipe_debug.cmd` (Windows) or the launcher in a terminal; read `~/bacpipe/app_state/app.log` |
| Page stays empty or the browser does not open | The address is in `app.log` ("serving on ..."); open it by hand |
| Pipeline fails | The terminal view shows the traceback; the complete output is in `~/bacpipe/app_state/runs/<time>/terminal_output.log`, and bacpipe's own logs in `<results>/<dataset>/logs` |
| "The dashboard could not be created" | The traceback is in the terminal view. Usually the same cause as when `visualize_using_dashboard` fails after `run_pipeline.py` |
| A setting is missing in the form | It is `hidden` in `form_overrides.yaml`, or the key is not in the YAML file of the installed bacpipe |
| Build: checksum mismatch | The `sha256` in `build_config.toml` does not belong to that version/release |
| Build: path too long | Step 7 prints the longest paths; add their folder to `[prune]` if it is not needed, otherwise raise `max_path_length` knowingly |
| Build: `No module named ziglang` (Windows) | `pip install -r app/requirements-build.txt` |
| Test `test_contract.py` fails | bacpipe changed something the application uses; the message says what to adapt |
| "Files missing" after unpacking on Windows | The folder was unpacked into a very deep location; unpack to a short path |
