"""
Check the GitHub API for a newer release of bacpipe. No user data is sent,
failures are ignored. Disabled by ``BACPIPE_NO_UPDATE_CHECK=1``.
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
