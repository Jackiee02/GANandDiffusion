import os
import sys
import time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# ── Configuration ───────────────────────────────────────────────────────────────
# Prevent xformers from attempting to load Triton
os.environ["XFORMERS_USE_TRITON"] = "0"

MODEL_ID = "stabilityai/stable-diffusion-2-1-base"

# Scene: solitary mountaineer at sunrise on a mountain peak

# Positive prompt (what we want the model to generate)
POSITIVE_PROMPT = (
    "(photorealistic:1.2), 8k, ultra detailed, "
    "(close-up:1.3), (medium close-up:1.2), portrait orientation, "
    "a solitary mountaineer filling most of the frame, telephoto lens, "
    "in winter gear standing on a rocky summit, "
    "overlooking a sea of clouds at sunrise, golden hour lighting, dramatic sky"
)

# Negative prompt (what we want the model to avoid)
NEGATIVE_PROMPT = (
    # Suppress duplicate or extra subjects
    "(multiple people:2), (crowd:2), (extra person:2), (duplicate person:2),\n"
    # Suppress extra limbs and malformed anatomy
    "(extra limbs:2), (extra arms:2), (extra legs:2), (fused fingers:2), (ghost limbs:2),\n"
    # Suppress multiple heads or torsos
    "(two heads:2), (multiple heads:2), (dual torso:2),\n"
    # Suppress low-quality output and deformations
    "(lowres:2), (blurry:2), (bad anatomy:2), (deformed:2), "
    "(poorly drawn face:2), (poorly drawn hands:2),\n"
    # Suppress compression artifacts, watermarks, and text
    "(jpeg artifacts:1.5), (artifact:1.5), (watermark:1.5), (text:1.5)"
)

OUTPUT_PATH    = "DiffusionPicture.png"
STEPS          = 150
HEIGHT         = 1024
WIDTH          = 1024
GUIDANCE_SCALE = 6.5


def setup_pipeline(model_id: str) -> StableDiffusionPipeline:
    """
    Load and configure the Stable Diffusion pipeline with:
      - FP16 precision and balanced device mapping
      - Attention slicing to reduce peak memory
      - Optional xformers memory-efficient attention
      - Optional model CPU offload (Linux only)
      - DPMSolverMultistepScheduler for crisper details
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="balanced",
        low_cpu_mem_usage=True,
    )
    pipe.enable_attention_slicing()

    # Try memory-efficient attention via xformers
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory-efficient attention enabled")
    except Exception:
        print("⚠️ xformers unavailable, skipping memory-efficient attention")

    # Enable CPU offload on non-Windows platforms
    if sys.platform != "win32":
        try:
            pipe.enable_model_cpu_offload()
            print("✅ model CPU offload enabled")
        except Exception:
            print("ℹ️ CPU offload not supported or failed, continuing on GPU/CPU")
    else:
        print("ℹ️ Windows detected: skipping CPU offload")

    # Use a stronger multi-step DPM solver scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def generate_image(
    positive_prompt: str,
    negative_prompt: str,
    output_path: str,
    model_id: str = MODEL_ID,
    steps: int = STEPS,
    height: int = HEIGHT,
    width: int = WIDTH,
    guidance_scale: float = GUIDANCE_SCALE,
):
    """
    Run the pipeline with the given prompts and parameters,
    then save the generated image to disk.
    """
    pipe = setup_pipeline(model_id)

    # Log where each component is loaded
    print(f"UNet on:         {pipe.unet.device}")
    print(f"Text encoder on: {pipe.text_encoder.device}")
    print(f"VAE on:          {pipe.vae.device}")

    # Perform inference
    start = time.time()
    image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
    ).images[0]

    # Wait for CUDA operations to finish (if applicable)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Inference time: {time.time() - start:.2f}s")

    # Save the result
    image.save(output_path)
    print(f"Saved → {output_path}")


def main():
    generate_image(POSITIVE_PROMPT, NEGATIVE_PROMPT, OUTPUT_PATH)


if __name__ == "__main__":
    main()