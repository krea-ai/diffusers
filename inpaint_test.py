# !pip install transformers accelerate
import os

import numpy as np
import torch

from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image
from mask_utils import create_gradient, expand_image, load_image
from masked2 import StableDiffusionMaskedImg2ImgPipeline


init_image = load_image(
    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
)
init_image = init_image.resize((512, 512))

# generator = torch.Generator(device="cpu").manual_seed(1)

mask_image = load_image(
    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
)
mask_image = mask_image.resize((512, 512))


img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)


img_path = "/home/erwann/diffusers/examples/community/new_image.png"
# mask_path = "/home/erwann/diffusers/examples/community/hard_mask_5.png"
mask_path = "/home/erwann/diffusers/examples/community/mask_image.png"
init_image = load_image(img_path)
mask_image = load_image(mask_path)
# mask_image.save("mask.png")


# new_width  = 480
# new_height = new_width * init_image.height / init_image.width 
# new_height = 640
# init_image = init_image.resize((new_width, int(new_height)))

# mask_image = mask_image.resize(init_image.size)
# mask_image = mask_image.resize((512, 512))

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.001] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


control_image = make_inpaint_condition(init_image, mask_image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)

# pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#     "/home/erwann/diffusers/examples/community/realistic_vision", controlnet=controlnet, torch_dtype=torch.float16
# )
from custom_inpaint_pipeline import StableDiffusionMaskedLatentControlNetInpaintPipeline


pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "/home/erwann/diffusers/examples/community/realistic_vision", controlnet=controlnet, torch_dtype=torch.float16
)

# pipe = StableDiffusionMaskedLatentControlNetInpaintPipeline.from_pretrained(
#     "/home/erwann/diffusers/examples/community/realistic_vision", controlnet=controlnet, torch_dtype=torch.float16
# )
pipe = StableDiffusionMaskedLatentControlNetInpaintPipeline(
    pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.unet, pipe.controlnet, pipe.scheduler, None, None,
)


# pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#     "/home/erwann/diffusers/examples/community/deliberate", controlnet=controlnet, torch_dtype=torch.float16
# )
# pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    # "/home/erwann/generation-service/safetensor-models/sd1.5", controlnet=controlnet, torch_dtype=torch.float16
# )
# generator = None
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


from diffusers import DPMSolverMultistepScheduler


pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)




# init_image = load_image("/home/erwann/diffusers/examples/community/castle.png")
init_image = load_image("/home/erwann/diffusers/examples/community/bmw.png")
init_image = init_image.resize((512, 512))


extended_image, mask_image = expand_image(init_image, expand_x=0, expand_y=-256)
print("Image size after extending, " + str(extended_image.size))
control_image = make_inpaint_condition(extended_image, mask_image)
blend_mask = create_gradient(mask_image, x=None, y=-256, offset=200)

extended_image.save("extended_image.png")
mask_image.save("mask_image.png")
blend_mask.save("blend_mask.png")
# vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").half()
# pipe.vae = vae

pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

generator = None
# generator = torch.Generator().manual_seed(456)
generator = torch.Generator().manual_seed(123)
# generate image
pipe.safety_checker = None
prompt= "bmw drifting, pink smoke"
images = pipe(
    prompt,
    num_inference_steps=25,
    generator=generator,
    guidance_scale=6.0,
    negative_prompt="deformed iris, deformed pupils, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
    # eta=1.0,
    # eta=1.0,
    # soft_mask=blend_mask,
    width=extended_image.width,
    height=extended_image.height,
    image=extended_image,
    mask_image=blend_mask,
    control_image=control_image,
    num_images_per_prompt=4,
    controlnet_conditioning_scale=1.,
    guess_mode=True,
).images


folder = "_".join(prompt.split(" ")) 
folder = "no_prompt" if len(folder) == 0 else folder
os.makedirs(folder, exist_ok=True)
print("Saving to ", folder)

for i, image in enumerate(images):
    image.save(os.path.join(folder, f"2_extend_{i}.png"))


#best config .35 / 20 steps

# img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("/home/erwann/generation-service/safetensor-models/real", safety_checker=None)
# img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("/home/erwann/generation-service/safetensor-models/realistic_vision", safety_checker=None)
img2img_pipe = StableDiffusionMaskedImg2ImgPipeline.from_pretrained("/home/erwann/generation-service/safetensor-models/realistic_vision", safety_checker=None)

print("Scheduler")
print(img2img_pipe.scheduler)


# img2imgpipe = StableDiffusionImg2ImgPipeline(
#     vae=pipe.vae,
#     text_encoder=pipe.text_encoder,
#     tokenizer=pipe.tokenizer,
#     unet=pipe.unet,
#     scheduler=pipe.scheduler,
#     safety_checker=None,
#     feature_extractor=pipe.feature_extractor,
# )


img2img_pipe = img2img_pipe.to("cuda")
img2img_pipe.enable_attention_slicing()
img2img_pipe.enable_xformers_memory_efficient_attention()

# soft_mask_pil = Image.open("/home/erwann/diffusers/examples/community/soft_mask_5.png")

# img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("/home/erwann/generation-service/safetensor-models/real", safety_checker=None)
from PIL import Image


for i, image in enumerate(images):
    final_image = img2img_pipe(
        prompt,
        image=image,
        mask_image=blend_mask,
        strength=0.350,
        num_inference_steps=19,
        generator=generator,
    ).images[0]
    final_image.save(os.path.join(folder, f"img2img_{i}_real_cfg8_9.png"))
    # plt.imshow(final_image)
    # plt.show()

# import matplotlib.pyplot as plt
# from PIL import Image



# plt.imshow(image)
# plt.show()
# plt.show()
