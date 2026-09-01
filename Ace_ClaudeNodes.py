"""
ComfyUI nodes for the Anthropic Files API workflow:

1. ClaudeListFiles   - GET /v1/files, outputs a text listing (id | name | size)
2. ClaudeFileRun     - POST /v1/messages with code_execution tool +
                       container_upload of a file_id; outputs response text
                       AND downloads any image URLs found in the response
                       as a ComfyUI IMAGE batch.

Zero external dependencies (urllib only; PIL/numpy/torch ship with ComfyUI).

The exact beta strings / tool type change over time. Defaults below are
editable node fields - copy the current values from Console Playground's
Code view if Anthropic rotates them.
"""

import io
import json
import re
import os
import urllib.request
import urllib.error

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
    h = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }
    if betas.strip():
        h["anthropic-beta"] = betas.strip()
    if workspace_id.strip():
        h["anthropic-workspace-id"] = workspace_id.strip()
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _key(api_key):
    return api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")


class ClaudeListFiles:
    CATEGORY = "Claude"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_list",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "workspace_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "betas": ("STRING", {"default": DEFAULT_BETAS}),
            },
        }

    def run(self, api_key, workspace_id, betas=DEFAULT_BETAS):
        key = _key(api_key)
        if not key:
            return ("ERROR: no API key",)
        req = urllib.request.Request(
            API_BASE + "/v1/files",
            headers=_headers(key, workspace_id, betas),
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return (f"ERROR: HTTP {e.code}: {e.read().decode('utf-8','replace')}",)
        except Exception as e:
            return (f"ERROR: {e}",)
        lines = [
            f"{f.get('id')} | {f.get('filename')} | {f.get('size_bytes')} bytes"
            for f in data.get("data", [])
        ]
        return ("\n".join(lines) if lines else "(no files)",)


class ClaudeFileRun:
    CATEGORY = "Claude"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("response", "images", "image_urls")

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
                "system": ("STRING", {"default": "", "multiline": True}),
                "betas": ("STRING", {"default": DEFAULT_BETAS}),
                "tool_type": ("STRING", {"default": DEFAULT_TOOL_TYPE}),
                "max_images": ("INT", {"default": 8, "min": 0, "max": 32}),
                "image_size": ("INT", {"default": 512, "min": 64, "max": 4096}),
            },
        }

    # ---- images -------------------------------------------------------

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
            # fit inside size x size canvas, pad with black, no distortion
            img.thumbnail((size, size), Image.LANCZOS)
            canvas = Image.new("RGB", (size, size), (0, 0, 0))
            canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
            used.append(url)
        if not tensors:
            return self._blank(size), used
        return torch.stack(tensors, dim=0), used

    # ---- main ---------------------------------------------------------

    def run(self, api_key, workspace_id, file_id, model, prompt, max_tokens,
            system="", betas=DEFAULT_BETAS, tool_type=DEFAULT_TOOL_TYPE,
            max_images=8, image_size=512):

        key = _key(api_key)
        if not key:
            return ("ERROR: no API key", self._blank(image_size), "")
        if not file_id.strip():
            return ("ERROR: file_id empty", self._blank(image_size), "")

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": [{"type": tool_type.strip(), "name": "code_execution"}],
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "container_upload", "file_id": file_id.strip()},
                    {"type": "text", "text": prompt},
                ],
            }],
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
            return (f"ERROR: HTTP {e.code}: {e.read().decode('utf-8','replace')}",
                    self._blank(image_size), "")
        except Exception as e:
            return (f"ERROR: request failed: {e}", self._blank(image_size), "")

        text = "".join(
            b.get("text", "") for b in data.get("content", [])
            if b.get("type") == "text"
        )

        urls = list(dict.fromkeys(IMG_URL_RE.findall(text)))  # dedupe, keep order
        images, used = self._download_images(urls, max_images, image_size)
        return (text, images, "\n".join(used))


NODE_CLASS_MAPPINGS = {
    "ClaudeListFiles": ClaudeListFiles,
    "ClaudeFileRun": ClaudeFileRun,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudeListFiles": "Claude: List Files",
    "ClaudeFileRun": "Claude: Run on File (+images)",
}
