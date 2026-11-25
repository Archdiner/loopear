import argparse
import io
from typing import Any

import numpy as np
import soundfile as sf
from aiohttp import web

from loopear.encoder import SR
from loopear.stream import LoopEarStream

INDEX_HTML: str = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LoopEar Local Prototype</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;background:#0e1116;color:#e6edf3}
button,select,input{padding:0.4rem 0.7rem;margin:0.2rem;border-radius:6px;border:1px solid #30363d;background:#161b22;color:#e6edf3}
pre{background:#161b22;padding:1rem;border-radius:8px;overflow:auto;border:1px solid #30363d}
.row{margin:0.75rem 0;display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center}
</style>
</head>
<body>
<h1>LoopEar Local Inference</h1>
<div class="row">
  <button id="micBtn">Start Mic</button>
  <label>Depth:
    <select id="depthSel">
      <option value="1">k=1</option>
      <option value="2" selected>k=2</option>
      <option value="4">k=4</option>
    </select>
  </label>
  <input id="wavUpload" type="file" accept="audio/*">
  <button id="feedWavBtn">Feed File</button>
</div>
<div class="row">
  <input id="enrollLabel" placeholder="Sound label">
  <input id="enrollFile" type="file" accept="audio/*">
  <button id="enrollBtn">Enroll</button>
</div>
<div class="row">
  <input id="cmpA" type="file" accept="audio/*">
  <input id="cmpB" type="file" accept="audio/*">
  <button id="cmpBtn">Compare A/B</button>
</div>
<pre id="out">Waiting for audio...</pre>
<script>
let ctx = null, proc = null, src = null, stream = null;
const out = document.getElementById("out");
async function refresh() {
  const r = await fetch("/api/state");
  out.textContent = JSON.stringify(await r.json(), null, 2);
}
document.getElementById("depthSel").onchange = async (e) => {
  await fetch("/api/depth", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({depth:Number(e.target.value)})});
  refresh();
};
document.getElementById("micBtn").onclick = async (e) => {
  if (ctx) { proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t=>t.stop()); ctx.close(); ctx=null; e.target.textContent="Start Mic"; return; }
  stream = await navigator.mediaDevices.getUserMedia({audio:true});
  ctx = new AudioContext({sampleRate:16000});
  src = ctx.createMediaStreamSource(stream);
  proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = async (ev) => {
    const ch = ev.inputBuffer.getChannelData(0);
    const r = await fetch("/api/pcm", {method:"POST", headers:{"Content-Type":"application/octet-stream"}, body:ch.buffer});
    out.textContent = JSON.stringify(await r.json(), null, 2);
  };
  src.connect(proc); proc.connect(ctx.destination);
  e.target.textContent = "Stop Mic";
};
document.getElementById("feedWavBtn").onclick = async () => {
  const f = document.getElementById("wavUpload").files[0]; if (!f) return;
  const r = await fetch("/api/pcm", {method:"POST", headers:{"Content-Type":"audio/wav"}, body:await f.arrayBuffer()});
  out.textContent = JSON.stringify(await r.json(), null, 2);
};
document.getElementById("enrollBtn").onclick = async () => {
  const lbl = document.getElementById("enrollLabel").value;
  const f = document.getElementById("enrollFile").files[0]; if (!lbl || !f) return;
  const r = await fetch("/api/enroll?label=" + encodeURIComponent(lbl), {method:"POST", headers:{"Content-Type":"audio/wav"}, body:await f.arrayBuffer()});
  out.textContent = JSON.stringify(await r.json(), null, 2);
};
document.getElementById("cmpBtn").onclick = async () => {
  const fa = document.getElementById("cmpA").files[0], fb = document.getElementById("cmpB").files[0]; if (!fa || !fb) return;
  const fd = new FormData(); fd.append("a", fa); fd.append("b", fb);
  const r = await fetch("/api/compare", {method:"POST", body:fd});
  out.textContent = JSON.stringify(await r.json(), null, 2);
};
refresh();
</script>
</body>
</html>
"""


def _decode_audio_bytes(raw: bytes, content_type: str = "") -> np.ndarray:
    if not raw:
        raise ValueError("Empty audio payload")
    ct = content_type.lower()
    if "wav" in ct or raw[:4] in (b"RIFF", b"fLaC", b"OggS"):
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != SR:
            t_old = np.linspace(0.0, 1.0, mono.shape[0], endpoint=False)
            n_new = max(1, int(round(mono.shape[0] * SR / sr)))
            t_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
            mono = np.interp(t_new, t_old, mono).astype(np.float32)
        return np.ascontiguousarray(mono, dtype=np.float32)
    if len(raw) % 4 != 0:
        raise ValueError("Raw float32 PCM payload byte length must be a multiple of 4")
    return np.ascontiguousarray(np.frombuffer(raw, dtype="<f4"), dtype=np.float32)


STREAM_KEY: web.AppKey[LoopEarStream] = web.AppKey("stream", LoopEarStream)


def create_app(stream: LoopEarStream) -> web.Application:
    app = web.Application(client_max_size=32 * 1024 * 1024)
    app[STREAM_KEY] = stream

    async def handle_index(_request: web.Request) -> web.Response:
        return web.Response(text=INDEX_HTML, content_type="text/html")

    async def handle_state(request: web.Request) -> web.Response:
        return web.json_response(request.app[STREAM_KEY].state())

    async def handle_depth(request: web.Request) -> web.Response:
        try:
            body: dict[str, Any] = await request.json()
            depth = int(body["depth"])
            new_depth = request.app[STREAM_KEY].set_depth(depth)
        except (KeyError, ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({"depth": new_depth})

    async def handle_pcm(request: web.Request) -> web.Response:
        try:
            if request.content_type == "application/json":
                body = await request.json()
                pcm = np.asarray(body["pcm"], dtype=np.float32)
            else:
                raw = await request.read()
                pcm = _decode_audio_bytes(raw, request.content_type)
            state = request.app[STREAM_KEY].feed(pcm)
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(state)

    async def handle_enroll(request: web.Request) -> web.Response:
        try:
            if request.content_type == "application/json":
                body = await request.json()
                label = str(body["label"])
                wav = np.asarray(body["pcm"], dtype=np.float32)
            else:
                label = request.query.get("label", "")
                raw = await request.read()
                wav = _decode_audio_bytes(raw, request.content_type)
            res = request.app[STREAM_KEY].enroll(label, wav)
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(res)

    async def handle_compare(request: web.Request) -> web.Response:
        try:
            if request.content_type == "application/json":
                body = await request.json()
                wav_a = np.asarray(body["pcm_a"], dtype=np.float32)
                wav_b = np.asarray(body["pcm_b"], dtype=np.float32)
                depth = body.get("depth")
            else:
                reader = await request.multipart()
                blobs: dict[str, bytes] = {}
                depth = None
                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    name = part.name or ""
                    data = await part.read()
                    if name == "depth":
                        depth = int(data.decode("utf-8"))
                    else:
                        blobs[name] = data
                wav_a = _decode_audio_bytes(blobs["a"], "audio/wav")
                wav_b = _decode_audio_bytes(blobs["b"], "audio/wav")
            res = request.app[STREAM_KEY].compare(
                wav_a, wav_b, k=int(depth) if depth is not None else None
            )
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(res)

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_state)
    app.router.add_post("/api/depth", handle_depth)
    app.router.add_post("/api/pcm", handle_pcm)
    app.router.add_post("/api/enroll", handle_enroll)
    app.router.add_post("/api/compare", handle_compare)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoopEar local web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    stream = LoopEarStream.from_checkpoint(
        ckpt=args.ckpt, depth=args.depth, device=args.device
    )
    web.run_app(create_app(stream), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
