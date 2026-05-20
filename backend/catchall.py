import json
from fastapi import Depends, Request, Response
import aiohttp

from main import get_http_session, app


# ---------------------------------------------------------------------------
# Ollama proxy (catch-all — must stay last to not shadow API routes)
# ---------------------------------------------------------------------------

def print_content(data: bytes, point_in_code: str):
    try:
        print(f"{point_in_code=} data=", json.dumps(json.loads(data.decode()), indent=2))
    except Exception:
        data_s = [d.strip() for d in data.decode().split("\n") if d.strip() != ""]
        for d in data_s:
            print(d)


@app.get("/{full_path:path}")
async def catch_all_get(
    full_path: str,
    response: Response,
    http_sess: aiohttp.ClientSession = Depends(get_http_session),
):
    async with http_sess.get("http://localhost:11434/" + full_path) as r:
        data = await r.content.read()
        print_content(data, "RESPONSE catch_all_get")
        response.status_code = r.status
        return data


@app.post("/{full_path:path}")
async def catch_all_post(
    full_path: str,
    request: Request,
    response: Response,
    http_sess: aiohttp.ClientSession = Depends(get_http_session),
):
    body = await request.body()
    print_content(body, "QUERY catch_all_post")
    async with http_sess.post("http://localhost:11434/" + full_path, data=body) as r:
        data = await r.content.read()
        print_content(data, "RESPONSE catch_all_post")
        response.status_code = r.status
        return data


@app.put("/{full_path:path}")
async def catch_all_put(
    full_path: str,
    request: Request,
    response: Response,
    http_sess: aiohttp.ClientSession = Depends(get_http_session),
):
    body = await request.body()
    print_content(body, "QUERY catch_all_put")
    async with http_sess.put("http://localhost:11434/" + full_path, data=body) as r:
        data = await r.content.read()
        response.status_code = r.status
        print_content(data, "RESPONSE catch_all_put")
        return data
