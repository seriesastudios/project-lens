"""Desktop shell for Project Lens.

Runs the FastAPI/uvicorn server in a background daemon thread and opens a
native macOS window (WKWebView via pywebview) pointed at it. Everything lives
in ONE process, so closing the window stops the server — nothing to orphan.

The local LLM (LM Studio) is intentionally NOT managed here; it stays external.

    ./venv/bin/python -m app.desktop      # attached (for debugging)
    ./lens                                # detached (frees the terminal)
"""
import socket
import threading
import time

import uvicorn
import webview


def _free_port() -> int:
    """Ask the OS for an unused loopback port so two instances never collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Server(uvicorn.Server):
    """uvicorn installs SIGINT/SIGTERM handlers that only work on the main
    thread; we run off-thread, so make that a no-op and drive shutdown via the
    `should_exit` flag instead."""

    def install_signal_handlers(self) -> None:  # noqa: D401 - override
        pass


def main() -> None:
    port = _free_port()
    config = uvicorn.Config(
        "app.main:app", host="127.0.0.1", port=port,
        reload=False, log_level="warning",
    )
    server = _Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for uvicorn to finish startup (lifespan: init_db + embedding backfill)
    # before pointing the window at it, so the first paint isn't a refused load.
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("Server thread exited before startup")
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("Server did not start within 30s")

    webview.create_window("Project Lens", f"http://127.0.0.1:{port}",
                          width=1100, height=800)
    webview.start()  # blocks until the window is closed

    # Window closed → ask the server to stop, then let the process exit (the
    # daemon thread dies with it regardless of how graceful that shutdown is).
    server.should_exit = True
    thread.join(timeout=5)


if __name__ == "__main__":
    main()
