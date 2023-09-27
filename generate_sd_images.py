import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


# set up stable diffusion
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


# begin script
M = 3*27 #images per YEAR
WHAT = "technology"
save_folder = "./SDimages"

for j in range(M):
    for YEAR in list(range(1900,2050)):
        N = 0
        while os.path.isfile(f"{save_folder}/{WHAT}{YEAR}_{N}.png"): N += 1
        if N<M:
            generator = torch.Generator("cuda").manual_seed(N*3000+YEAR)
            prompt = f"{WHAT} of the year {YEAR}"
            image = pipe(prompt, generator = generator).images[0]
            image.save(f"{save_folder}/{WHAT}{YEAR}_{N}.png")
