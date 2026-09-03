"""
ACE Claude nodes. Install: copy the whole ACE_Claude_Nodes/ folder into
ComfyUI/custom_nodes/ so it looks like:
    custom_nodes/ACE_Claude_Nodes/__init__.py
    custom_nodes/ACE_Claude_Nodes/ace_claude_nodes.py
    custom_nodes/ACE_Claude_Nodes/web/ace_claude_upload.js
Restart ComfyUI and hard-refresh the browser (Ctrl+Shift+R).

ACE_Claude_List_Files - GET /v1/files, outputs "id | name | size" text
ACE_Claude_Push_File  - "choose file to upload" button on the node: pick ANY
                        file type, it is held in RAM only (never written to
                        disk) and pushed to the Anthropic Files API when the
                        node runs. Outputs the file_id.
ACE_Claude_File_Node  - POST /v1/messages with code_execution +
                        container_upload; optional reference_image is sent
                        as an image block with the prompt; downloads image
                        URLs found in the response as an IMAGE batch.

Zero external dependencies (urllib only; PIL/numpy/torch/aiohttp ship with
ComfyUI).
"""

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

API_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_BETAS = "files-api-2025-04-14,code-execution-2025-08-25"
DEFAULT_TOOL_TYPE = "code_execution_20250825"

MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
]

IMG_URL_RE = re.compile(r"https?://[^\s\"'<>()\[\]]+?\.(?:jpg|jpeg|png|webp|gif)", re.I)

# In-memory store for browser uploads: {filename: bytes}. RAM only, nothing
# on disk, cleared when ComfyUI stops. Same filename re-uploaded = replaced.
_PENDING_FILES = {}

_NO_FILES_PLACEHOLDER = "(none - use the upload button)"


def _headers(api_key, workspace_id, betas, json_body=False):
    h = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    if betas.strip():
        h["anthropic-beta"] = betas.strip()
    if workspace_id.strip():
        h["anthropic-workspace-id"] = workspace_id.strip()
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _key(api_key):
    return api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")


def _image_to_b64_png(image_tensor):
    """ComfyUI IMAGE tensor (B,H,W,C float 0-1) -> base64 PNG of first frame."""
    import numpy as np
    from PIL import Image

    arr = image_tensor[0].cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

class ClaudeDeleteFile:
    CATEGORY = "ACE_Claude_Nodes"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("deleted_id", "raw_json")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "workspace_id": ("STRING", {"default": ""}),
                "file_id": ("STRING", {"default": ""}),
                "confirm_delete": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "betas": ("STRING", {"default": DEFAULT_BETAS}),
            },
        }

    def run(self, api_key, workspace_id, file_id, confirm_delete, betas=DEFAULT_BETAS):
        key = _key(api_key)
        if not key:
            return ("", "ERROR: no API key")
        if not file_id.strip():
            return ("", "ERROR: file_id empty")
        if not confirm_delete:
            return ("", "SKIPPED: set confirm_delete to true. Deletion is permanent, no undo.")
        req = urllib.request.Request(
            API_BASE + "/v1/files/" + file_id.strip(),
            headers=_headers(key, workspace_id, betas),
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return ("", f"ERROR: HTTP {e.code}: {e.read().decode('utf-8','replace')}")
        except Exception as e:
            return ("", f"ERROR: {e}")
        return (data.get("id", ""), json.dumps(data, indent=2, ensure_ascii=False))

class ClaudeListFiles:
    CATEGORY = "ACE_Claude_Nodes"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_list", "file_id", "filename")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "workspace_id": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999}),
            },
            "optional": {
                "betas": ("STRING", {"default": DEFAULT_BETAS}),
            },
        }

    def run(self, api_key, workspace_id, index, betas=DEFAULT_BETAS):
        key = _key(api_key)
        if not key:
            return ("ERROR: no API key", "", "")
        req = urllib.request.Request(
            API_BASE + "/v1/files",
            headers=_headers(key, workspace_id, betas),
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return (f"ERROR: HTTP {e.code}: {e.read().decode('utf-8','replace')}", "", "")
        except Exception as e:
            return (f"ERROR: {e}", "", "")
        files = data.get("data", [])
        lines = [
            f"{i}: {f.get('id')} | {f.get('filename')} | {f.get('size_bytes')} bytes"
            for i, f in enumerate(files)
        ]
        listing = "\n".join(lines) if lines else "(no files)"
        if 0 <= index < len(files):
            sel = files[index]
            return (listing, sel.get("id", ""), sel.get("filename", ""))
        return (listing, "", "")


class ClaudePushFile:
    CATEGORY = "ACE_Claude_Nodes"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("file_id", "filename", "raw_json")

    @classmethod
    def INPUT_TYPES(cls):
        files = sorted(_PENDING_FILES) or [_NO_FILES_PLACEHOLDER]
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "workspace_id": ("STRING", {"default": ""}),
                "file": (files,),
            },
            "optional": {
                "betas": ("STRING", {"default": DEFAULT_BETAS}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, file, **kwargs):
        # Always accept here; run() gives a precise error. Strict combo
        # validation would reject files uploaded after the graph was built.
        return True

    def run(self, api_key, workspace_id, file, betas=DEFAULT_BETAS):
        import mimetypes

        key = _key(api_key)
        if not key:
            return ("", "", "ERROR: no API key")
        if file not in _PENDING_FILES:
            return ("", "",
                    f"ERROR: no uploaded file selected. Open /ace_claude/upload "
                    f"in a browser tab, upload a file, press R in ComfyUI, then "
                    f"pick it in the dropdown. (got: {file!r})")

        payload = _PENDING_FILES[file]
        mime = mimetypes.guess_type(file)[0] or "application/octet-stream"

        boundary = "----ACEClaudeBoundary" + os.urandom(16).hex()
        body = b"".join([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{file}"\r\n'.encode("utf-8"),
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
            payload,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ])

        headers = _headers(key, workspace_id, betas)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = urllib.request.Request(
            API_BASE + "/v1/files",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            return ("", "", f"ERROR: HTTP {e.code}: {raw}")
        except Exception as e:
            return ("", "", f"ERROR: {e}")
        return (
            data.get("id", ""),
            data.get("filename", ""),
            json.dumps(data, indent=2, ensure_ascii=False),
        )


class ClaudeFileRun:
    CATEGORY = "ACE_Claude_Nodes"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("response", "images", "image_urls", "raw_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "workspace_id": ("STRING", {"default": ""}),
                "file_id": ("STRING", {"default": ""}),
                "model": (MODELS, {"default": "claude-sonnet-4-6"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 128000}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "system": ("STRING", {"default": "", "multiline": True}),
                "betas": ("STRING", {"default": DEFAULT_BETAS}),
                "tool_type": ("STRING", {"default": DEFAULT_TOOL_TYPE}),
                "max_images": ("INT", {"default": 8, "min": 0, "max": 32}),
                "image_size": ("INT", {"default": 512, "min": 64, "max": 4096}),
            },
        }

    def _blank(self, size):
        import torch
        return torch.zeros((1, size, size, 3), dtype=torch.float32)

    def _download_images(self, urls, max_images, size):
        import numpy as np
        import torch
        from PIL import Image

        tensors, used = [], []
        for url in urls[:max_images]:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                continue
            img.thumbnail((size, size), Image.LANCZOS)
            canvas = Image.new("RGB", (size, size), (0, 0, 0))
            canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
            used.append(url)
        if not tensors:
            return self._blank(size), used
        return torch.stack(tensors, dim=0), used

    def run(self, api_key, workspace_id, file_id, model, prompt, max_tokens,
            reference_image=None, system="", betas=DEFAULT_BETAS,
            tool_type=DEFAULT_TOOL_TYPE, max_images=8, image_size=512):

        key = _key(api_key)
        if not key:
            return ("ERROR: no API key", self._blank(image_size), "", "")
        if not file_id.strip():
            return ("ERROR: file_id empty", self._blank(image_size), "", "")

        content = [{"type": "container_upload", "file_id": file_id.strip()}]
        if reference_image is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _image_to_b64_png(reference_image),
                },
            })
        content.append({"type": "text", "text": prompt})

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": [{"type": tool_type.strip(), "name": "code_execution"}],
            "messages": [{"role": "user", "content": content}],
        }
        if system.strip():
            body["system"] = system

        req = urllib.request.Request(
            API_BASE + "/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers=_headers(key, workspace_id, betas, json_body=True),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            return (f"ERROR: HTTP {e.code}: {raw}", self._blank(image_size), "", raw)
        except Exception as e:
            return (f"ERROR: request failed: {e}", self._blank(image_size), "", "")

        text = "".join(
            b.get("text", "") for b in data.get("content", [])
            if b.get("type") == "text"
        )

        stop = data.get("stop_reason")
        if stop and stop != "end_turn":
            text += f"\n\n[stop_reason: {stop}]"

        urls = list(dict.fromkeys(IMG_URL_RE.findall(text)))
        images, used = self._download_images(urls, max_images, image_size)
        return (text, images, "\n".join(used),
                json.dumps(data, indent=2, ensure_ascii=False))


_UPLOAD_HTML = """<!doctype html>
<meta charset="utf-8">
<title>ACE Claude upload</title>
<h3>Upload any file (held in memory, sent to Claude by the Push File node)</h3>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <button>Upload</button>
</form>
<pre>{files}</pre>
"""


def _register_server_routes():
    """Built-in upload page at /ace_claude/upload. Uploaded bytes are kept in
    RAM only; nothing is written to disk. No-op outside a running ComfyUI."""
    try:
        from aiohttp import web
        from server import PromptServer
        if getattr(PromptServer.instance, "_ace_claude_routes_registered", False):
            return
        PromptServer.instance._ace_claude_routes_registered = True
        routes = PromptServer.instance.routes
    except Exception:
        return

    def _page():
        listing = "\n".join(
            f"{name}  ({len(data)} bytes, in memory)"
            for name, data in sorted(_PENDING_FILES.items())
        ) or "(no files uploaded yet)"
        return _UPLOAD_HTML.replace("{files}", listing)

    @routes.get("/ace_claude/upload")
    async def _ace_upload_page(request):
        return web.Response(text=_page(), content_type="text/html")

    @routes.post("/ace_claude/upload")
    async def _ace_upload_post(request):
        post = await request.post()
        f = post.get("file")
        if f is None or not getattr(f, "file", None):
            return web.Response(status=400, text="no file")
        name = os.path.basename(f.filename or "").strip()
        if not name:
            return web.Response(status=400, text="bad filename")
        _PENDING_FILES[name] = f.file.read()
        return web.Response(
            text=f'stored in memory: "{name}" ({len(_PENDING_FILES[name])} bytes)\n'
                 f"In ComfyUI press R (refresh node definitions), then pick it in "
                 f"the ACE Claude: Push File dropdown and run the node."
        )


_register_server_routes()

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "ACE_Claude_List_Files": ClaudeListFiles,
    "ACE_Claude_Push_File": ClaudePushFile,
    "ACE_Claude_File_Node": ClaudeFileRun,
    "ACE_Claude_Delete_File": ClaudeDeleteFile,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_Claude_List_Files": "ACE Claude: List Files",
    "ACE_Claude_Push_File": "ACE Claude: Push File",
    "ACE_Claude_File_Node": "ACE Claude: Run on File (+images)",
    "ACE_Claude_Delete_File": "ACE Claude: Delete File"

}
