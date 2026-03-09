"""
generate_video.py — Example: generate a video from a text prompt via ComfyUI MCP.

This example shows how an AI agent would call the ComfyUI pipeline end-to-end:
  1. Connect to a local ComfyUI instance
  2. Submit a text prompt
  3. Poll until the generation is complete
  4. Print the output file path

Usage:
    python examples/generate_video.py

Prerequisites:
    - ComfyUI running on localhost:8188
    - A compatible model installed (AnimateDiff + SD1.5 recommended)
    - pip install -r requirements.txt
"""

import asyncio
import os
import sys
import time

# Add the project root to path so we can import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.comfyui_client import ComfyUIClient


# ── Configuration ─────────────────────────────────────────────────────────────

COMFYUI_HOST = os.getenv("COMFYUI_HOST", "localhost")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))

# The prompt to generate — change this to whatever you want
DEFAULT_PROMPT = (
    "a lone astronaut walking on Mars, "
    "cinematic lighting, "
    "dust storms in background, "
    "photorealistic, 4K"
)

DEFAULT_NEGATIVE = "ugly, blurry, low quality, watermark, text"


# ── Main example ──────────────────────────────────────────────────────────────

async def generate_video(prompt: str, negative_prompt: str = DEFAULT_NEGATIVE) -> str:
    """
    Submit a generation job and wait for it to complete.

    This is the pattern your agent would use:
      - One call in, one file path out
      - Error bubbles up naturally as exceptions

    Returns:
        Path to the generated output file.
    """
    print(f"Connecting to ComfyUI at {COMFYUI_HOST}:{COMFYUI_PORT}...")

    async with ComfyUIClient(host=COMFYUI_HOST, port=COMFYUI_PORT) as client:

        # Quick health check — see what's in the queue
        queue = await client.get_queue_status()
        print(f"Queue status: {queue['queue_running']} running, {queue['queue_pending']} pending")

        if queue["queue_pending"] > 5:
            print("Warning: queue is deep — this may take a while")

        # Submit the generation job
        print(f"\nPrompt: {prompt[:80]}...")
        print("Submitting job...", end=" ", flush=True)
        start = time.time()

        job_id = await client.queue_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        print(f"Job ID: {job_id}")

        # Poll until done
        print("Waiting for generation", end="", flush=True)
        while True:
            history = await client._get_history(job_id)
            if history:
                break
            print(".", end="", flush=True)
            await asyncio.sleep(3)

        elapsed = time.time() - start
        print(f" done! ({elapsed:.1f}s)")

        # Extract output path
        output_path = client._extract_output_path(history)
        return output_path


async def main():
    """Run the example."""
    prompt = DEFAULT_PROMPT

    # Allow passing a custom prompt via command line
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    try:
        output = await generate_video(prompt)
        print(f"\n✓ Generation complete!")
        print(f"  Output: {output}")
        print(f"  Find the file in your ComfyUI output directory.")
        print()
        print("Next steps:")
        print("  - See src/comfyui_client.py for the full client API")
        print("  - Read the full tutorial: https://dev.to/clawgear")
        print("  - Star the repo if this was useful: https://github.com/pinzasai/comfyui-agent-pipeline")

    except ConnectionRefusedError:
        print("\n✗ Could not connect to ComfyUI.")
        print("  Make sure ComfyUI is running: python main.py (from your ComfyUI directory)")
        print(f"  Expected at: http://{COMFYUI_HOST}:{COMFYUI_PORT}")
        sys.exit(1)

    except TimeoutError as e:
        print(f"\n✗ Timed out: {e}")
        print("  The job was submitted but didn't complete in time.")
        print("  Check ComfyUI's console for errors.")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
