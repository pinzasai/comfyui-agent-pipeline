# comfyui-agent-pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)

> **Production-grade Python pipeline for AI agents to generate video via the ComfyUI MCP server.**

Connect Claude (or any LLM agent) to a local ComfyUI instance and generate video from text prompts — with error handling, retry logic, queue management, and output verification.

---

## What this is

This repo is the companion code for the **[ComfyUI Agent Backend tutorial series](https://dev.to/clawgear)** on Dev.to.

The series covers:
- **Part 1:** Why your AI agent needs a local video generation backend (cost math: RunwayML = ~$1,500/month vs ComfyUI = ~$6/month)
- **Part 2:** How to connect Claude to ComfyUI via MCP in 5 minutes
- **Part 3:** Building a production-grade pipeline (this repo)

---

## Requirements

- Python 3.10+
- ComfyUI running on `localhost:8188`
- ComfyUI MCP server installed (`npm install -g comfyui-mcp-server` or `pip install comfyui-mcp-server`)
- A video model installed in ComfyUI (AnimateDiff, CogVideoX, or Wan recommended)
- GPU with 8GB+ VRAM

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/pinzasai/comfyui-agent-pipeline.git
cd comfyui-agent-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure ComfyUI is running on port 8188

# 4. Run the example
python examples/generate_video.py
```

---

## Usage

```python
import asyncio
from src.comfyui_client import ComfyUIClient

async def main():
    async with ComfyUIClient(host="localhost", port=8188) as client:
        job_id = await client.queue_prompt(
            prompt="a lone astronaut walking on Mars, cinematic lighting, dust storms in background"
        )
        output = await client.get_output(job_id)
        print(f"Video ready: {output}")

asyncio.run(main())
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `host` | `localhost` | ComfyUI host |
| `port` | `8188` | ComfyUI port |
| `max_queue_depth` | `5` | Reject jobs if queue exceeds this |
| `job_timeout_seconds` | `600` | Max wait time per job |
| `max_retries` | `3` | Retry attempts on transient failure |
| `max_concurrent_jobs` | `3` | Parallel job limit |

---

## Workflow Templates

Included in `examples/`:
- `animatediff_basic` — AnimateDiff + SD1.5, 5-second clips
- `cogvideox` — CogVideoX-2B, higher quality
- `wan_t2v` — Wan 2.1 14B, best quality, slower

---

## Error Handling

| Error | Meaning | What the agent can do |
|---|---|---|
| Queue at capacity | Too many concurrent jobs | Wait 2–3 minutes and retry |
| Job timeout | GPU OOM or model crash | Retry with simpler prompt or shorter duration |
| Output corrupted | Disk full or write error | Check disk space on generation server |
| CUDA error | GPU out of memory | Reduce resolution or clip length |
| MCP connection refused | ComfyUI not running | Alert user: backend offline |

---

## Benchmarks

| Hardware | Model | Clip Length | Time |
|---|---|---|---|
| MacBook Pro M3 Max (36GB) | CogVideoX-2B | 5s 480p | ~90s |
| RTX 4090 (24GB) | AnimateDiff | 5s 512×512 | ~35s |
| 2× RTX 3090 | Wan 2.1 14B | 10s 720p | ~180s |

---

## Contributing

PRs welcome. Most wanted:
- New workflow templates for Wan 2.1, CogVideoX-5B, HunyuanVideo
- Better error messages from ComfyUI internals
- Test coverage for edge cases (queue full, GPU OOM, disk full)

---

## License

MIT
