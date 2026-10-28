#!/usr/bin/env python3
"""Local OpenAI-compatible proxy between graphrag 3.x and the DeepSeek API.

    python deepseek_proxy.py --port 18801 --stats outputs/hotpotqa/proxy_stats.json

graphrag's own code and prompts are not modified. Every chat request is forwarded to DeepSeek
unchanged except for three rewrites, which apply the settings every other system in
evaluation_scripts uses (common/env.sh):

  1. thinking   `thinking: {"type": "disabled"}` is added (DeepSeek V4/V4.1 think by default),
                unless LLM_DISABLE_THINKING=0.
  2. temperature  any `temperature` is dropped, so DeepSeek uses its default (1.0), unless
                LLM_TEMPERATURE is set, in which case that value is sent on every call.
                graphrag 3.2 indexing sends no temperature; this is a safeguard.
  3. JSON       `response_format: {"type": "json_schema", ...}` becomes `{"type": "json_object"}`.
                graphrag 3.x asks for a JSON schema only in create_community_reports; DeepSeek rejects
                json_schema with HTTP 400 (microsoft/graphrag#2200). The stock community-report prompt
                already spells out the JSON fields, and graphrag still validates every reply against its
                pydantic model (CommunityReportResponse).

The Authorization header is passed through (graphrag sends the DeepSeek key from settings.yaml), so the
proxy holds no secret. Counts of each rewrite and of every HTTP status are written to --stats.
"""
import argparse
import asyncio
import json
import os
import signal
from collections import Counter

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

FORWARD_HEADERS = ("authorization", "content-type", "accept")


def rewrite(body: dict, stats: Counter) -> dict:
    stats["requests"] += 1
    if os.environ.get("LLM_DISABLE_THINKING", "1") != "0":
        body["thinking"] = {"type": "disabled"}
        stats["thinking_disabled"] += 1
    temperature = os.environ.get("LLM_TEMPERATURE") or None
    if temperature is not None:
        body["temperature"] = float(temperature)
        stats["temperature_set"] += 1
    elif "temperature" in body:
        del body["temperature"]
        stats["temperature_dropped"] += 1
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        body["response_format"] = {"type": "json_object"}
        stats["json_schema_to_json_object"] += 1
    return body


def make_app(upstream: str, stats: Counter, stats_path: str | None) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)

    async def on_startup(app):
        # DeepSeek holds queued requests open for up to 10 minutes before inference starts
        app["session"] = ClientSession(connector=TCPConnector(limit=0), timeout=ClientTimeout(total=1800))

    async def on_cleanup(app):
        await app["session"].close()
        dump_stats(stats, stats_path)

    async def chat(request: web.Request) -> web.StreamResponse:
        body = rewrite(await request.json(), stats)
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_HEADERS}
        async with app["session"].post(f"{upstream}/chat/completions", json=body, headers=headers) as up:
            stats[f"http_{up.status}"] += 1
            if body.get("stream") and up.status == 200:
                resp = web.StreamResponse(status=up.status, headers={"Content-Type": up.headers.get("Content-Type", "text/event-stream")})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
            data = await up.read()
            return web.Response(status=up.status, body=data, content_type="application/json")

    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/chat/completions", chat)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def dump_stats(stats: Counter, path: str | None):
    if path:
        with open(path, "w") as f:
            json.dump(dict(sorted(stats.items())), f, indent=2)


async def periodic_dump(stats: Counter, path: str | None):
    while True:
        await asyncio.sleep(30)
        dump_stats(stats, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/"))
    ap.add_argument("--stats", default=None, help="JSON file with rewrite / status counts (updated every 30 s)")
    args = ap.parse_args()

    stats: Counter = Counter()

    async def run():
        runner = web.AppRunner(make_app(args.upstream, stats, args.stats))
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", args.port, backlog=4096).start()
        print(f"deepseek_proxy: 127.0.0.1:{args.port} -> {args.upstream}", flush=True)
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        dumper = asyncio.create_task(periodic_dump(stats, args.stats))
        await stop.wait()
        dumper.cancel()
        await runner.cleanup()

    asyncio.run(run())


if __name__ == "__main__":
    main()
