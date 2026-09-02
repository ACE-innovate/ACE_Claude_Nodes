"""
ACE Claude nodes - single file, drop into ComfyUI/custom_nodes/ and restart.

ACE_Claude_List_Files - GET /v1/files, outputs "id | name | size" text
ACE_Claude_File_Node  - POST /v1/messages with code_execution +
                        container_upload; optional reference_image is sent
                        as an image block with the prompt; downloads image
                        URLs found in the response as an IMAGE batch.

Zero external dependencies (urllib only; PIL/numpy/torch ship with ComfyUI).
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


NODE_CLASS_MAPPINGS = {
    "ACE_Claude_List_Files": ClaudeListFiles,
    "ACE_Claude_File_Node": ClaudeFileRun,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE_Claude_List_Files": "ACE Claude: List Files",
    "ACE_Claude_File_Node": "ACE Claude: Run on File (+images)",
}
