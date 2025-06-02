import torch
from diffusers import FluxPipeline, StableDiffusionPipeline, EulerDiscreteScheduler
from io import BytesIO

'''
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
'''

model_id = "stabilityai/stable-diffusion-2-1-base"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.enable_attention_slicing()

#save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
#pipe.enable_model_cpu_offload() 

def generate_image(text_prompt: str) -> tuple[bytes, str]:

    image = pipe(prompt=text_prompt).images[0]  

    # Get image bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    image_bytes = img_byte_arr.getvalue()

    return image_bytes
