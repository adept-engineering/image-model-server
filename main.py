import torch
from diffusers import FluxPipeline
from io import BytesIO

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# Get image bytes
img_byte_arr = BytesIO()
image.save(img_byte_arr, format='PNG')
image_bytes = img_byte_arr.getvalue()

# Get base64
import base64
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Save to file
image.save("flux-dev.png")
