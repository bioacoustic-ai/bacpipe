"""
Build the bacpipe application for the system this script runs on.

    python app/build.py

The result is a folder that contains everything needed to run bacpipe, plus
an archive of that folder to hand out:

    app/dist/bacpipe/                       (Windows, Linux)
        bacpipe.exe  |  bacpipe             the launcher
        runtime/                            Python with bacpipe installed
        README.txt, LICENSE
    app/dist/bacpipe.app/                   (macOS)
    app/dist/bacpipe-<version>-<system>-<arch>.zip | .tar.gz

How it works - there is no "freezing" of the code involved. The build takes
a relocatable Python, installs bacpipe into it with pip exactly like a user
would, and adds a small launcher that starts `python -m bacpipe_app`. So if
`pip install` gives a working bacpipe, the application works too, and a
change in bacpipe needs no change in this script.

The steps, in the order of the functions below:

    1. download_python      fetch the pinned standalone Python (cached)
    2. install_packages     pip install the dependencies and bacpipe
    3. install_app          copy bacpipe_app, the README and build_info.json
    4. prune                delete development files that are not needed
    5. compile_bytecode     precompile .pyc files for a faster first start
    6. make_launcher        bacpipe.exe / bacpipe / bacpipe.app
    7. check_paths          refuse paths that are too long for Windows
    8. make_archive         zip (Windows) or tar.gz (Linux, macOS)

Requirements for running this script: Python >= 3.11 with the packages of
app/requirements-build.txt, and internet access. Options: `--help`.
Everything that is a choice (Python version, package list, ...) is in
app/build_config.toml. The documentation is in app/README.md.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
LAUNCHER_DIR = APP_DIR / "launcher"
CACHE_DIR = APP_DIR / ".cache"
DEFAULT_OUTPUT = APP_DIR / "dist"

PYTHON_DOWNLOAD_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    "{release}/cpython-{version}+{release}-{triple}-install_only_stripped.tar.gz"
)


# --------------------------------------------------------------------
# What is built where
# --------------------------------------------------------------------
def current_system():
    """'windows', 'macos' or 'linux'."""
    return {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}[platform.system()]


def current_arch():
    """'x86_64' or 'aarch64'."""
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def python_triple(system, arch):
    """Platform name used in the file names of python-build-standalone."""
    return {
        "windows": f"{arch}-pc-windows-msvc",
        "macos": f"{arch}-apple-darwin",
        "linux": f"{arch}-unknown-linux-gnu",
    }[system]


@dataclass
class Layout:
    """The places inside the application folder."""

    system: str
    root: Path  # the folder that is shipped
    runtime: Path  # the Python installation
    launcher: Path  # what the user starts
    info_dir: Path  # where README.txt and LICENSE go

    @property
    def python(self):
        """The console Python of the runtime."""
        if self.system == "windows":
            return self.runtime / "python.exe"
        return self.runtime / "bin" / "python3"


def layout_for(output_dir, system=None, name="bacpipe"):
    """
    The layout of the application below ``output_dir``.

    Also used by app/tests/test_build.py to find the parts of a build.
    """
    system = system or current_system()
    output_dir = Path(output_dir)
    if system == "macos":
        root = output_dir / f"{name}.app"
        resources = root / "Contents" / "Resources"
        return Layout(
            system=system,
            root=root,
            runtime=resources / "runtime",
            launcher=root / "Contents" / "MacOS" / name,
            info_dir=resources,
        )
    root = output_dir / name
    return Layout(
        system=system,
        root=root,
        runtime=root / "runtime",
        launcher=root / (f"{name}.exe" if system == "windows" else name),
        info_dir=root,
    )


def load_config():
    """app/build_config.toml as a dict."""
    return tomllib.loads((APP_DIR / "build_config.toml").read_text(encoding="utf-8"))


def project_version():
    """The version of bacpipe: the single source is pyproject.toml."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def step(title):
    print(f"\n=== {title} ===", flush=True)


def run(command, **kwargs):
    """Run a command, show it, and stop the build if it fails."""
    print("  $ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def git_commit():
    """(commit hash, has uncommitted changes) - ('', False) without git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "", False


def site_packages(layout):
    """The site-packages folder of the runtime (asks the runtime itself)."""
    result = subprocess.run(
        [str(layout.python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def folder_size(folder):
    return sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file() and not f.is_symlink())


# --------------------------------------------------------------------
# 1. The Python runtime
# --------------------------------------------------------------------
def download_python(config, system, arch):
    """
    Download the pinned standalone Python into app/.cache (once).

    Returns
    -------
    pathlib.Path
        the downloaded archive
    """
    step("1/8 Python runtime")
    triple = python_triple(system, arch)
    url = PYTHON_DOWNLOAD_URL.format(
        release=config["python"]["release"], version=config["python"]["version"], triple=triple
    )
    archive = CACHE_DIR / url.rsplit("/", 1)[1]
    if not archive.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {url}")
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(archive)
    else:
        print(f"  using cached {archive.name}")

    expected = config["python"].get("sha256", {}).get(triple)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if expected and actual != expected:
        archive.unlink()
        raise SystemExit(
            f"Checksum mismatch for {archive.name}:\n  expected {expected}\n  got      {actual}\n"
            "The download was removed. Check [python.sha256] in build_config.toml."
        )
    if not expected:
        print(f"  WARNING: no sha256 for {triple} in build_config.toml (got {actual})")
    return archive


def unpack_python(archive, layout):
    """Unpack the Python archive into the runtime folder of the application."""
    if layout.root.exists():
        shutil.rmtree(layout.root)
    layout.runtime.parent.mkdir(parents=True, exist_ok=True)
    unpack_dir = layout.runtime.parent / "_unpack"
    with tarfile.open(archive) as tar:
        tar.extractall(unpack_dir, filter="tar")
    # the archive contains one top level folder called "python"
    (unpack_dir / "python").rename(layout.runtime)
    unpack_dir.rmdir()
    run([layout.python, "--version"])


# --------------------------------------------------------------------
# 2. bacpipe and its dependencies
# --------------------------------------------------------------------
def pip_install(layout, installer, link_mode, *arguments):
    """Install into the runtime with uv (fast) or pip."""
    if installer == "uv":
        run(
            [
                "uv", "pip", "install",
                "--python", layout.python,
                "--link-mode", link_mode,
                # The requirements file adds the PyTorch package index
                # (--extra-index-url). That index also hosts old copies of
                # ordinary packages like tqdm. pip picks the best version
                # from all indexes; uv by default only looks at the first
                # index that has the package and then fails on the version
                # bounds. This option makes uv behave like pip, which is
                # what the test workflows install with.
                "--index-strategy", "unsafe-best-match",
                *arguments,
            ]
        )
    else:
        run(
            [
                layout.python, "-m", "pip", "install",
                "--no-warn-script-location", "--disable-pip-version-check",
                *arguments,
            ]
        )


def install_packages(config, layout, requirements, installer, link_mode):
    """
    Install bacpipe into the runtime, the same way a user would.

    Parameters
    ----------
    requirements : str
        "pyproject" or the name of a requirements file in the repository,
        see [requirements] in build_config.toml
    """
    step(f"2/8 Packages ({requirements}, installed with {installer})")
    if requirements == "pyproject":
        pip_install(layout, installer, link_mode, REPO_ROOT)
    else:
        requirements_file = REPO_ROOT / requirements
        if not requirements_file.exists():
            raise SystemExit(f"Requirements file not found: {requirements_file}")
        # run from the repository root, relative paths inside the
        # requirements file refer to it
        pip_install(layout, installer, link_mode, "-r", requirements_file)
        pip_install(layout, installer, link_mode, "--no-deps", REPO_ROOT)

    extra = config.get("extra_requirements", {}).get(layout.system, [])
    if extra:
        pip_install(layout, installer, link_mode, *extra)


# --------------------------------------------------------------------
# 3. The application package
# --------------------------------------------------------------------
def install_app(config, layout, requirements):
    """
    Copy ``bacpipe_app`` into the runtime, together with the README for the
    help page and ``build_info.json``.
    """
    step("3/8 Application package")
    target = site_packages(layout) / "bacpipe_app"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        APP_DIR / "bacpipe_app", target, ignore=shutil.ignore_patterns("__pycache__", "resources")
    )

    # the README and the images it shows, for the help page
    sys.path.insert(0, str(APP_DIR))
    from bacpipe_app.readme_links import local_images

    readme_dir = target / "resources" / "readme"
    readme_dir.mkdir(parents=True)
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    (readme_dir / "README.md").write_text(readme_text, encoding="utf-8")
    for image in local_images(readme_text, REPO_ROOT):
        (readme_dir / image).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / image, readme_dir / image)
        print(f"  help page image: {image}")

    # the drawing on the "Play the bacpipe" button
    shutil.copy2(REPO_ROOT / "src" / "bacpipe.svg", target / "resources" / "bacpipe.svg")

    commit, dirty = git_commit()
    info = {
        "version": project_version(),
        "commit": commit + ("+local-changes" if dirty else ""),
        "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "system": layout.system,
        "arch": current_arch(),
        "python": config["python"]["version"],
        "requirements": requirements,
    }
    (target / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("  build_info.json: " + json.dumps(info))

    for name in ("LICENSE",):
        shutil.copy2(REPO_ROOT / name, layout.info_dir / name)
    readme_for_users = LAUNCHER_DIR / f"README_{layout.system}.txt"
    text = readme_for_users.read_text(encoding="utf-8").replace("@VERSION@", info["version"])
    newline = "\r\n" if layout.system == "windows" else "\n"
    with open(layout.info_dir / "README.txt", "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
    return info


# --------------------------------------------------------------------
# 4. Remove what is not needed
# --------------------------------------------------------------------
def prune(config, layout):
    """Delete the files matching [prune] patterns in build_config.toml."""
    step("4/8 Prune")
    packages = site_packages(layout)
    before = folder_size(layout.root)
    for pattern in config.get("prune", {}).get("patterns", []):
        matches = list(packages.glob(pattern))
        for match in matches:
            if match.is_dir() and not match.is_symlink():
                shutil.rmtree(match)
            elif match.exists() or match.is_symlink():
                match.unlink()
        print(f"  {pattern}: removed {len(matches)}")
    saved = (before - folder_size(layout.root)) / 1e6
    print(f"  saved {saved:.0f} MB")


# --------------------------------------------------------------------
# 5. Bytecode
# --------------------------------------------------------------------
def compile_bytecode(layout):
    """
    Precompile all python files of the runtime.

    Python otherwise does this on first import, which makes the first start
    slow and fails silently if the application folder is read-only.
    """
    step("5/8 Compile bytecode")
    # some packages ship files that are not valid for this Python version
    # (templates, python 2 examples); they do not matter, so errors are
    # not fatal here
    subprocess.run(
        [str(layout.python), "-m", "compileall", "-q", "-j", "0", str(site_packages(layout))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  done")


# --------------------------------------------------------------------
# 6. Launcher and icon
# --------------------------------------------------------------------
def make_icon(config, target, sizes):
    """Create an .ico / .icns / .png icon from the image in the config."""
    from PIL import Image

    with Image.open(REPO_ROOT / config["app"]["icon"]) as image:
        image = image.convert("RGBA")
        largest = max(sizes)
        big = image.resize((largest, largest), Image.LANCZOS)
        if target.suffix == ".png":
            big.save(target)
        else:
            big.save(target, sizes=[(size, size) for size in sizes])
    return target


def make_windows_launcher(config, layout, version):
    """
    Compile launcher.c into bacpipe.exe with icon, manifest and version
    information (launcher.rc), using the zig compiler of the "ziglang"
    pip package.
    """
    work_dir = CACHE_DIR / "launcher_build"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(LAUNCHER_DIR / "windows", work_dir)
    make_icon(config, work_dir / "bacpipe.ico", sizes=[16, 24, 32, 48, 64, 256])

    numbers = [part for part in version.split(".") if part.isdigit()][:3]
    numbers += ["0"] * (4 - len(numbers))
    resource_script = (work_dir / "launcher.rc.in").read_text(encoding="utf-8")
    for placeholder, value in {
        "ICON_FILE": "bacpipe.ico",
        "VERSION_COMMAS": ",".join(numbers),
        "VERSION": version,
        "COMPANY": config["app"]["company"],
        "DESCRIPTION": config["app"]["description"],
        "COPYRIGHT": f"{config['app']['company']}, see LICENSE",
    }.items():
        resource_script = resource_script.replace(f"@{placeholder}@", value)
    (work_dir / "launcher.rc").write_text(resource_script, encoding="utf-8")

    run(
        [
            sys.executable, "-m", "ziglang", "cc",
            "launcher.c", "launcher.rc",
            "-o", layout.launcher,
            "-target", config["windows"]["zig_target"],
            "-municode",  # entry point wWinMain
            "-Wl,--subsystem,windows",  # no console window
            "-O2", "-s",
        ],
        cwd=work_dir,
    )
    # zig writes debug information next to the exe, it is not needed
    for leftover in layout.launcher.parent.glob(layout.launcher.stem + ".pdb"):
        leftover.unlink()

    # the console variant; .cmd files need Windows line endings
    script = (LAUNCHER_DIR / "windows" / "bacpipe_debug.cmd").read_text(encoding="utf-8")
    with open(layout.root / "bacpipe_debug.cmd", "w", encoding="utf-8", newline="\r\n") as f:
        f.write(script)
    shutil.copy2(work_dir / "bacpipe.ico", layout.root / "bacpipe.ico")


def make_unix_launcher(layout):
    """Write the shell script that starts the runtime."""
    runtime = os.path.relpath(layout.runtime, layout.launcher.parent)
    script = (LAUNCHER_DIR / "unix" / "bacpipe.sh.in").read_text(encoding="utf-8")
    layout.launcher.parent.mkdir(parents=True, exist_ok=True)
    with open(layout.launcher, "w", encoding="utf-8", newline="\n") as f:
        f.write(script.replace("@RUNTIME@", runtime))
    layout.launcher.chmod(0o755)


def make_macos_bundle(config, layout, version):
    """Info.plist and icon that turn the folder into bacpipe.app."""
    name = config["app"]["name"]
    try:
        make_icon(config, layout.info_dir / f"{name}.icns", sizes=[16, 32, 64, 128, 256, 512])
    except Exception as error:
        print(f"  WARNING: could not create the .icns icon: {error}")
    info = {
        "CFBundleName": name,
        "CFBundleDisplayName": name,
        "CFBundleIdentifier": f"ai.bioacoustic.{name}",
        "CFBundleExecutable": layout.launcher.name,
        "CFBundleIconFile": f"{name}.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "bacpipe plays and analyses audio.",
    }
    with open(layout.root / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(info, f)


def make_launcher(config, layout, version):
    step("6/8 Launcher")
    if layout.system == "windows":
        make_windows_launcher(config, layout, version)
    else:
        make_unix_launcher(layout)
        if layout.system == "macos":
            make_macos_bundle(config, layout, version)
        else:
            make_icon(config, layout.root / "bacpipe.png", sizes=[256])
    print(f"  {layout.launcher}")


# --------------------------------------------------------------------
# 7. Path lengths
# --------------------------------------------------------------------
def longest_paths(layout, count=5):
    """The longest paths relative to the parent of the application folder."""
    base = layout.root.parent
    lengths = sorted(
        ((len(str(f.relative_to(base))), str(f.relative_to(base))) for f in layout.root.rglob("*")),
        reverse=True,
    )
    return lengths[:count]


def check_paths(config, layout):
    """
    Windows refuses paths longer than 260 characters unless an administrator
    changed a system setting. The folder has to work wherever a user unpacks
    it, so the longest path inside the folder is limited.
    """
    step("7/8 Path lengths")
    limit = config["windows"]["max_path_length"]
    longest = longest_paths(layout)
    for length, path in longest:
        print(f"  {length:4d}  {path}")
    if layout.system == "windows" and longest and longest[0][0] > limit:
        raise SystemExit(
            f"The longest path has {longest[0][0]} characters, the limit is {limit} "
            "([windows] max_path_length in build_config.toml).\n"
            "Add the folder of the offending files to [prune] patterns if they "
            "are not needed at runtime."
        )


# --------------------------------------------------------------------
# 8. The archive
# --------------------------------------------------------------------
def make_archive(layout, version, output_dir):
    """
    Pack the application folder: .zip on Windows (the Explorer can unpack
    it), .tar.gz elsewhere (keeps symlinks and the executable flags).
    """
    step("8/8 Archive")
    base_name = f"bacpipe-{version}-{layout.system}-{current_arch()}"
    started = time.time()
    if layout.system == "windows":
        archive = output_dir / f"{base_name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for file in sorted(layout.root.rglob("*")):
                zf.write(file, file.relative_to(layout.root.parent))
    else:
        archive = output_dir / f"{base_name}.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
            tar.add(layout.root, arcname=layout.root.name)
    print(f"  {archive}  ({archive.stat().st_size / 1e6:.0f} MB, {time.time() - started:.0f} s)")
    return archive


# --------------------------------------------------------------------
def main():
    config = load_config()
    system, arch = current_system(), current_arch()

    parser = argparse.ArgumentParser(
        description="Build the bacpipe application for this system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--requirements",
        default=config["requirements"][system],
        help='"pyproject" or a requirements file in the repository root',
    )
    parser.add_argument(
        "--installer",
        choices=["auto", "uv", "pip"],
        default="auto",
        help="auto = uv if it is installed, otherwise pip",
    )
    parser.add_argument(
        "--link-mode",
        default="copy",
        choices=["copy", "hardlink", "clone", "symlink"],
        help="uv only: how packages get from the uv cache into the runtime. "
        '"hardlink" saves disk space for local test builds.',
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output folder")
    parser.add_argument("--no-compile", action="store_true", help="skip step 5 (bytecode)")
    parser.add_argument("--no-archive", action="store_true", help="skip step 8 (archive)")
    args = parser.parse_args()

    installer = args.installer
    if installer == "auto":
        installer = "uv" if shutil.which("uv") else "pip"

    version = project_version()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = layout_for(output_dir, system, config["app"]["name"])
    print(f"Building bacpipe {version} for {system} {arch} -> {layout.root}")
    started = time.time()

    archive = download_python(config, system, arch)
    unpack_python(archive, layout)
    install_packages(config, layout, args.requirements, installer, args.link_mode)
    install_app(config, layout, args.requirements)
    prune(config, layout)
    if not args.no_compile:
        compile_bytecode(layout)
    make_launcher(config, layout, version)
    check_paths(config, layout)
    packed = None if args.no_archive else make_archive(layout, version, output_dir)

    print(
        f"\nFinished in {(time.time() - started) / 60:.1f} min. "
        f"Application folder: {layout.root} ({folder_size(layout.root) / 1e9:.2f} GB)"
    )
    if packed:
        print(f"Archive: {packed}")
    print(f"Test it with:  python -m pytest app/tests --app-dist {output_dir}")


if __name__ == "__main__":
    main()
