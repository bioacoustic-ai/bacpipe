"""
Tell the user when a newer version of the application has been released.

On start of a *built* application one small request is sent to the GitHub
API (latest release of the bacpipe repository). No data about the user is
transmitted, and without a connection the check is skipped silently.
Set the environment variable ``BACPIPE_NO_UPDATE_CHECK=1`` to disable it.
"""

import json
import os
import re
import urllib.request

from . import REPO_URL

LATEST_RELEASE_API = REPO_URL.replace(
    "https://github.com/", "https://api.github.com/repos/"
) + "/releases/latest"


def parse_version(text):
    """'v1.3.5' -> (1, 3, 5); anything unparsable -> ()."""
    numbers = re.findall(r"\d+", text.split("+")[0])
    return tuple(int(n) for n in numbers[:3])


def is_newer(latest, current):
    """True if version string ``latest`` is newer than ``current``."""
    latest_parts, current_parts = parse_version(latest), parse_version(current)
    return bool(latest_parts and current_parts and latest_parts > current_parts)


def update_notice(current_version, timeout=4):
    """
    Returns
    -------
    str
        markdown text for the sidebar, empty if there is nothing to say
    """
    if os.environ.get("BACPIPE_NO_UPDATE_CHECK"):
        return ""
    try:
        request = urllib.request.Request(
            LATEST_RELEASE_API, headers={"User-Agent": "bacpipe-app"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            release = json.load(response)
        if is_newer(release["tag_name"], current_version):
            return (
                f"Version {release['tag_name'].lstrip('v')} is available: "
                f"[download]({release['html_url']})"
            )
    except Exception:
        pass
    return ""
