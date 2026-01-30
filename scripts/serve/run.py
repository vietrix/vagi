from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vAGI serving API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    uvicorn.run(
        "serve.app.main:createApp",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    main()
