"""
The links inside README.md.

Used by the help page (to make the links work outside of the repository) and
by ``app/build.py`` (to find the images that have to be shipped with the
README). Standard library only.
"""

import re
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

# markdown:  ![alt](target)  /  [text](target "title")
_MD_LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
# html:  src="target"  /  href="target"
_HTML_LINK = re.compile(r"""((?:src|href)\s*=\s*["'])([^"']+)(["'])""")


def _is_relative(target):
    """True for 'src/logo.png', False for 'https://...', '#anchor', '/abs'."""
    return not re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*:|#|/)", target)


def _file_part(target):
    """'./src/logo.png?raw=true#x' -> 'src/logo.png'"""
    file_part = target.split("#")[0].split("?")[0]
    while file_part.startswith("./"):
        file_part = file_part[2:]
    return file_part


def _is_local_image(target, root):
    file_part = _file_part(target)
    local = Path(root) / file_part
    return (
        bool(file_part)
        and ".." not in Path(file_part).parts
        and local.suffix.lower() in IMAGE_SUFFIXES
        and local.is_file()
    )


def local_images(markdown_text, root):
    """
    Relative paths of the images the README shows that exist below ``root``.

    Returns
    -------
    list of str
        e.g. ``["src/bacpipe_logo.png", "src/bacpipe_demo.gif"]``
    """
    targets = [m.group(2) for m in _MD_LINK.finditer(markdown_text)]
    targets += [m.group(2) for m in _HTML_LINK.finditer(markdown_text)]
    found = []
    for target in targets:
        if _is_relative(target) and _is_local_image(target, root):
            if _file_part(target) not in found:
                found.append(_file_part(target))
    return found


# an address written as plain text, not already part of a link or tag
_BARE_URL = re.compile(r"""(?<!\]\()(?<![\w<"'=\]/])(https?://[^\s<>()\[\]"']+)""")


def link_bare_urls(markdown_text):
    """
    Turn addresses written as plain text into links, like GitHub does when
    it shows the README ("... under this link: https://example.org.").
    """

    def replace(match):
        url = match.group(1)
        trailing = ""
        while url and url[-1] in ".,;:!?":  # punctuation of the sentence
            url, trailing = url[:-1], url[-1] + trailing
        return f"<{url}>{trailing}"

    return _BARE_URL.sub(replace, markdown_text)


def rewrite_links(markdown_text, root, repo_url):
    """
    Make the relative links of the README work outside of the repository.

    Images that exist below ``root`` are served by the application itself
    (``readme_assets/...``), every other relative link is pointed to the
    file on GitHub.
    """

    def replace(match):
        prefix, target, suffix = match.groups()
        if not _is_relative(target):
            return match.group(0)
        if _is_local_image(target, root):
            new_target = "readme_assets/" + _file_part(target)
        else:
            new_target = f"{repo_url}/blob/main/{target.lstrip('./')}"
        return prefix + new_target + suffix

    text = _MD_LINK.sub(replace, markdown_text)
    return _HTML_LINK.sub(replace, text)
