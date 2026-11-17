"""
Native file and folder dialogs, run as a separate process because a dialog
needs the main thread:

    python dialogs.py folder [start_directory]
    python dialogs.py file   [start_directory]

Prints the chosen path, or nothing if the dialog was cancelled. Uses tkinter
on Windows and macOS, and ``zenity`` or ``kdialog`` on Linux with tkinter as
fallback. Standard library only, must not import bacpipe_app.
"""

import os
import shutil
import subprocess
import sys

TITLES = {
    "folder": "bacpipe - select a folder",
    "file": "bacpipe - select the annotations file",
}


def _run(command):
    """Output of a dialog program; '' if it was cancelled."""
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def ask_with_zenity(mode, start_dir):
    command = ["zenity", "--file-selection", f"--title={TITLES[mode]}"]
    if mode == "folder":
        command.append("--directory")
    else:
        command += [
            "--file-filter=Tables (csv, tsv, txt) | *.csv *.tsv *.txt",
            "--file-filter=All files | *",
        ]
    if start_dir:
        # the trailing separator makes zenity open the folder itself
        command.append(f"--filename={start_dir.rstrip(os.sep)}{os.sep}")
    return _run(command)


def ask_with_kdialog(mode, start_dir):
    start = start_dir or os.path.expanduser("~")
    if mode == "folder":
        command = ["kdialog", "--title", TITLES[mode], "--getexistingdirectory", start]
    else:
        command = ["kdialog", "--title", TITLES[mode], "--getopenfilename", start]
    return _run(command)


def ask_with_tkinter(mode, start_dir):
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    # the browser has the focus, make sure the dialog appears in front of it
    root.attributes("-topmost", True)
    root.update()
    if mode == "folder":
        chosen = filedialog.askdirectory(
            parent=root, initialdir=start_dir or None, title=TITLES[mode], mustexist=True
        )
    else:
        chosen = filedialog.askopenfilename(
            parent=root,
            initialdir=start_dir or None,
            title=TITLES[mode],
            filetypes=[("Tables", "*.csv *.tsv *.txt"), ("All files", "*")],
        )
    root.destroy()
    return chosen or ""


def ask(mode, start_dir=""):
    """
    Parameters
    ----------
    mode : str
        "folder" or "file"
    start_dir : str
        the folder the dialog opens in

    Returns
    -------
    str
        the chosen path, '' if the dialog was cancelled
    """
    if sys.platform.startswith("linux"):
        if shutil.which("zenity"):
            return ask_with_zenity(mode, start_dir)
        if shutil.which("kdialog"):
            return ask_with_kdialog(mode, start_dir)
    return ask_with_tkinter(mode, start_dir)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in TITLES:
        start = sys.argv[2] if len(sys.argv) > 2 else ""
        try:
            sys.stdout.write(ask(sys.argv[1], start))
        except Exception as error:  # no dialog available / no display
            sys.stderr.write(f"dialog not available: {error}\n")
            sys.exit(1)
