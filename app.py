from __future__ import annotations

from core.config import configure_page_once, render_shell_css, render_shell_header
from core.navigation import render_bottom_navigation, render_current_page, render_sidebar


def main() -> None:
    configure_page_once()
    render_shell_css()
    model_name, page_name = render_sidebar()
    render_shell_header(model_name, page_name)
    render_current_page(model_name, page_name)
    render_bottom_navigation()


if __name__ == "__main__":
    main()
