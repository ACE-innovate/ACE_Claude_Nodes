# ACE Claude Nodes

ComfyUI custom nodes for the Anthropic Files API and Messages API with code execution. Upload any file to Claude, run a prompt against it, and pull generated images back into your workflow.

## Nodes

| Node | Purpose |
|---|---|
| **ACE Claude: List Files** | Lists all files in your Anthropic workspace (`GET /v1/files`). Outputs the listing as text, plus the `file_id` and `filename` of the entry selected by `index`. |
| **ACE Claude: Push File** | Adds a "choose file to upload" button to the node. Any file type. The file is held in server RAM only (never written to disk) and pushed to the Anthropic Files API when the node runs. Outputs the `file_id`. |
| **ACE Claude: Run on File (+images)** | Sends a `POST /v1/messages` request with the code execution tool and a `container_upload` of the given `file_id`. Optional `reference_image` input is sent as an image block with the prompt. Image URLs found in the response are downloaded and output as an IMAGE batch. |

## Installation

Copy the folder into `ComfyUI/custom_nodes/` so it looks like:

```
custom_nodes/ace_claude_nodes/__init__.py
custom_nodes/ace_claude_nodes/ace_claude_nodes.py
custom_nodes/ace_claude_nodes/web/ace_claude_upload.js
```

The Python file must be named exactly `ace_claude_nodes.py` (lowercase) — `__init__.py` imports it by that name and Linux filesystems are case-sensitive.

Restart ComfyUI and hard-refresh the browser (Ctrl+Shift+R).

No external dependencies: uses only `urllib` plus PIL/numpy/torch/aiohttp, which ship with ComfyUI.

## API key

Either paste your key into the `api_key` field on each node, or leave it blank and set the `ANTHROPIC_API_KEY` environment variable before starting ComfyUI. `workspace_id` is optional; leave blank to use the key's default workspace.

## Usage

### Upload and run

1. Add **ACE Claude: Push File**, click **choose file to upload**, pick a file. It's stored in memory on the ComfyUI server.
2. Wire its `file_id` output into **ACE Claude: Run on File (+images)**.
3. Set model, prompt, and optionally a `reference_image`, `system` prompt, `max_images`, and `image_size`.
4. Queue the workflow. The Push File node uploads to the Files API, the Run node sends the message with code execution enabled and returns the text response, any downloaded images as an IMAGE batch, the image URLs, and the raw JSON.

Alternative upload page: open `http://<comfyui-host>:8188/ace_claude/upload` in a browser tab, upload a file, press **R** in ComfyUI to refresh node definitions, then pick the file in the Push File dropdown.

### Reuse an existing file

Use **ACE Claude: List Files** to browse workspace files; set `index` to select one and wire its `file_id` output into the Run node.

## Notes

- Uploaded files are held in RAM only and cleared when ComfyUI stops. Re-uploading the same filename replaces the previous bytes.
- Files pushed to the Anthropic Files API persist in your workspace until deleted (`DELETE /v1/files/{file_id}`). They are visible to any API key with access to that workspace.
- Image download: URLs matching common image extensions (jpg, jpeg, png, webp, gif) in the response text are fetched, letterboxed onto a square canvas of `image_size`, and stacked into a batch. If none download, a black placeholder image is output.
- If the response ends for a reason other than `end_turn` (e.g. `max_tokens`), the stop reason is appended to the text output.
- Default betas: `files-api-2025-04-14,code-execution-2025-08-25`. Default tool type: `code_execution_20250825`. Both are editable on the nodes.

## Models

Dropdown includes: `claude-sonnet-4-6` (default), `claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5`, `claude-fable-5`.
