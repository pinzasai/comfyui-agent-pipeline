"""
comfyui_client.py — Minimal async HTTP client for the ComfyUI API.

Covers the two core operations an agent needs:
  - queue_prompt(prompt) → job_id
  - get_output(job_id)   → file path (polls until done)

Usage:
    async with ComfyUIClient() as client:
        job_id = await client.queue_prompt("a cat on the moon")
        output = await client.get_output(job_id)
        print(output)
"""

import asyncio
import json
import uuid
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Base workflow template — AnimateDiff with SD 1.5
# Swap in your own workflow JSON as needed.
# ---------------------------------------------------------------------------
BASE_WORKFLOW = {
    "3": {
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 7.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSampler"
    },
    "4": {
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
        "class_type": "CheckpointLoaderSimple"
    },
    "5": {
        "inputs": {"width": 512, "height": 512, "batch_size": 16},
        "class_type": "EmptyLatentImage"
    },
    "6": {
        "inputs": {"text": "PROMPT_PLACEHOLDER", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode"
    },
    "7": {
        "inputs": {"text": "ugly, blurry, low quality", "clip": ["4", 1]},
        "class_type": "CLIPTextEncode"
    },
    "8": {
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        "class_type": "VAEDecode"
    },
    "9": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0]
        },
        "class_type": "SaveImage"
    }
}


class ComfyUIClient:
    """
    Async HTTP client for the ComfyUI API.

    Designed as a context manager — handles session lifecycle automatically.

        async with ComfyUIClient(host="localhost", port=8188) as client:
            job_id = await client.queue_prompt("a futuristic city at night")
            path   = await client.get_output(job_id)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8188,
        poll_interval: float = 3.0,
        timeout_seconds: float = 600.0,
    ):
        self.base_url = f"http://{host}:{port}"
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
        self._client_id = str(uuid.uuid4())

    # ── Context manager ──────────────────────────────────────────────────────

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    # ── Public API ───────────────────────────────────────────────────────────

    async def queue_prompt(
        self,
        prompt: str,
        negative_prompt: str = "ugly, blurry, low quality",
        workflow: Optional[dict] = None,
    ) -> str:
        """
        Submit a generation job to ComfyUI.

        Args:
            prompt: Text description of the video/image to generate.
            negative_prompt: What to exclude from the generation.
            workflow: Optional custom workflow dict. If None, uses BASE_WORKFLOW.

        Returns:
            prompt_id (str): Job identifier for polling.
        """
        wf = json.loads(json.dumps(workflow or BASE_WORKFLOW))  # deep copy

        # Inject prompts into the workflow
        for node_id, node in wf.items():
            if node.get("class_type") == "CLIPTextEncode":
                text = node["inputs"].get("text", "")
                if "PROMPT_PLACEHOLDER" in text or (
                    text == "ugly, blurry, low quality" and negative_prompt
                ):
                    if "PROMPT_PLACEHOLDER" in text:
                        node["inputs"]["text"] = prompt
                    else:
                        node["inputs"]["text"] = negative_prompt

        payload = {
            "prompt": wf,
            "client_id": self._client_id,
        }
        resp = await self._client.post("/prompt", json=payload)
        resp.raise_for_status()
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ValueError(f"No prompt_id in response: {data}")
        return prompt_id

    async def get_output(self, prompt_id: str) -> str:
        """
        Poll until the job is complete, then return the output file path.

        Args:
            prompt_id: Job ID returned by queue_prompt().

        Returns:
            Filename of the generated output (e.g. "ComfyUI_00001_.png").

        Raises:
            TimeoutError: If the job doesn't complete within timeout_seconds.
            RuntimeError: If the job fails.
        """
        elapsed = 0.0
        while elapsed < self.timeout_seconds:
            history = await self._get_history(prompt_id)
            if history:
                return self._extract_output_path(history)
            await asyncio.sleep(self.poll_interval)
            elapsed += self.poll_interval

        raise TimeoutError(
            f"Job {prompt_id} did not complete within {self.timeout_seconds}s"
        )

    async def get_queue_status(self) -> dict:
        """Return current queue depth (running + pending jobs)."""
        resp = await self._client.get("/queue")
        resp.raise_for_status()
        data = resp.json()
        return {
            "queue_running": len(data.get("queue_running", [])),
            "queue_pending": len(data.get("queue_pending", [])),
        }

    async def get_system_stats(self) -> dict:
        """Return GPU/CPU/memory stats from ComfyUI."""
        resp = await self._client.get("/system_stats")
        resp.raise_for_status()
        return resp.json()

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _get_history(self, prompt_id: str) -> Optional[dict]:
        """Return the history entry for this job, or None if not done."""
        resp = await self._client.get(f"/history/{prompt_id}")
        resp.raise_for_status()
        data = resp.json()
        entry = data.get(prompt_id)
        if entry and entry.get("outputs"):
            return entry["outputs"]
        return None

    @staticmethod
    def _extract_output_path(outputs: dict) -> str:
        """Pull the first output filename from a history outputs dict."""
        for node_outputs in outputs.values():
            for category in node_outputs.values():
                if isinstance(category, list) and category:
                    item = category[0]
                    if isinstance(item, dict) and "filename" in item:
                        return item["filename"]
        raise RuntimeError(f"No output file found in history: {outputs}")
