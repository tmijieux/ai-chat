import uvicorn
import sys

if __name__ == "__main__":
    # On Windows, uvicorn's asyncio loop factory returns SelectorEventLoop when
    # running in a reload subprocess, which doesn't support create_subprocess_exec.
    # We pass a custom loop factory via module:attribute string — uvicorn pickles
    # config.loop into the worker process and calls import_from_string() there,
    # so the factory runs in the child and returns ProactorEventLoop.
    loop = "uvicorn_patch:proactor_loop_factory" if sys.platform == "win32" else "asyncio"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, loop=loop)
