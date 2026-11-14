"""
The help page: the README of bacpipe as a web page under ``/help``.

The README is converted from markdown to HTML when the page is opened. Its
images are served from ``/readme_assets/``. Links to other files of the
repository are pointed to GitHub, because those files are not part of the
application.
"""

import html
from pathlib import Path

from tornado.web import HTTPError, RequestHandler, StaticFileHandler

from . import REPO_URL, paths
from .readme_links import IMAGE_SUFFIXES, link_bare_urls, rewrite_links

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>bacpipe help</title>
<link rel="icon" href="readme_assets/__favicon__">
<style>
  body {{ margin: 0; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         color: #1f2328; line-height: 1.55; }}
  header {{ background: #0072b5; color: white; padding: 14px 28px; font-size: 1.3em;
            font-weight: 600; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 20px 28px 80px; }}
  img {{ max-width: 100%; }}
  a {{ color: #0969da; }}
  h1, h2 {{ border-bottom: 1px solid #d8dee4; padding-bottom: 0.3em; }}
  code {{ background: #eff1f3; padding: 0.15em 0.35em; border-radius: 5px; font-size: 90%; }}
  pre {{ background: #f6f8fa; padding: 14px; border-radius: 6px; overflow-x: auto; }}
  pre code {{ background: none; padding: 0; }}
  table {{ border-collapse: collapse; display: block; overflow-x: auto; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 12px; }}
  blockquote {{ border-left: 4px solid #d0d7de; margin-left: 0; padding-left: 16px; color: #59636e; }}
</style>
</head>
<body>
<header>bacpipe help</header>
<main>
{body}
</main>
</body>
</html>
"""


def render_help_page():
    """The complete HTML of the help page."""
    root = paths.readme_root()
    if root is None:
        body = (
            "<p>The README file is not part of this installation. You can "
            f'read it online: <a href="{REPO_URL}">{html.escape(REPO_URL)}</a></p>'
        )
        return PAGE.format(body=body)

    import markdown

    text = (root / "README.md").read_text(encoding="utf-8")
    body = markdown.markdown(
        link_bare_urls(rewrite_links(text, root, REPO_URL)),
        # tables, fenced code blocks, heading ids for the links of the
        # table of contents
        extensions=["extra", "toc", "sane_lists"],
    )
    return PAGE.format(body=body)


class HelpHandler(RequestHandler):
    """Serves the help page."""

    def get(self):
        self.set_header("Content-Type", "text/html; charset=utf-8")
        self.write(render_help_page())


class ReadmeAssetHandler(StaticFileHandler):
    """Serves the images of the README (and nothing but images)."""

    def initialize(self, **kwargs):
        root = paths.readme_root()
        super().initialize(path=str(root) if root else str(paths.PACKAGE_DIR))

    def parse_url_path(self, url_path):
        if url_path == "__favicon__":
            return url_path
        if Path(url_path).suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPError(404)
        return super().parse_url_path(url_path)

    @classmethod
    def get_absolute_path(cls, root, path):
        if path == "__favicon__":
            return str(paths.favicon_path())
        return super().get_absolute_path(root, path)

    def validate_absolute_path(self, root, absolute_path):
        if absolute_path == str(paths.favicon_path()):
            return absolute_path
        return super().validate_absolute_path(root, absolute_path)
