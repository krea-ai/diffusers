# !pip install transformers accelerate
import os
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
from contexttimer import Timer
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image
from mask_utils import create_gradient, expand_image, make_inpaint_condition


def controlnet_inpaint(
    self,
    prompt,
    image: Union[str, PIL.Image],
    mask: Union[str, PIL.Image],
    output_type="pil",
    pipe=None,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    seed=None,
):
    """
    Takes image and mask (can be PIL images, a local path, or a URL)
    """
    init_image = load_image(image)
    mask_image = load_image(mask)
    control_image = make_inpaint_condition(init_image, mask_image)

    with Timer() as t:
        if pipe is None:
            # pipe = sd.get_pipeline()
            raise NotImplementedError("Need to pass in a pipeline for now")
    t_load = t.elapsed

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    with Timer() as t:
        images = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
            negative_prompt="deformed iris, deformed pupils, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
            width=init_image.width,
            height=init_image.height,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            num_images_per_prompt=4,
            output_type=output_type,
        ).images
    t_inference = t.elapsed

    return {
        "images": images,
        "performance": {
            "t_load": t_load,
            "t_inference": t_inference,
        }
    }

def controlnet_extend(
    self,
    prompt: Union[str, List[str]],
    init_image: Union[torch.FloatTensor, PIL.Image.Image],
    model_id: str = "realistic_vision",
    seed: Optional[int] = None,
    expand_offset_x: int = 0,
    expand_offset_y: int = 0,
    img2img_strength: float = 0.35,
    img2img_steps: int = 15,
    mask_offset: float = 40,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    output_type: Optional[str] = "pil",
    # pipe
):

    with Timer() as t:
        controlnet_inpaint_pipe = self.model_loader.get_pipeline(model_id, StableDiffusionControlNetInpaintPipeline)
    t_load = t.elapsed

    with Timer() as t:
        extended_image, mask_img = expand_image(init_image, expand_x=expand_offset_x, expand_y=expand_offset_y)
        print("Image size after extending, " + str(extended_image.size))

        blend_mask = create_gradient(mask_img, x=expand_offset_x, y=expand_offset_y, offset=mask_offset)
    t_preprocess = t.elapsed
    
    inpaint_results = controlnet_inpaint(
        prompt,
        extended_image,
        mask_img,
        output_type="pil",
        pipe=controlnet_inpaint_pipe,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
    )
    inpainted_images = inpaint_results["images"]
    performance = inpaint_results["performance"]


    img2img_pipe = StableDiffusionImg2ImgPipeline(
        vae=controlnet_inpaint_pipe.vae,
        text_encoder=controlnet_inpaint_pipe.text_encoder,
        tokenizer=controlnet_inpaint_pipe.tokenizer,
        unet=controlnet_inpaint_pipe.unet,
        scheduler=controlnet_inpaint_pipe.scheduler,
        safety_checker=None,
        feature_extractor=controlnet_inpaint_pipe.feature_extractor,
    )
    img2img_pipe = img2img_pipe.to("cuda")
    

    # uses masked img2img to homogenize the inpainted zone and the original image, making them look more natural and reduce border artefacts
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    with Timer() as t:
        final_images = img2img_pipe(
            prompt,
            # negative_prompt=negative_prompt,
            negative_prompt=None,
            image=inpainted_images,
            mask_image=blend_mask,
            strength=img2img_strength,
            num_inference_steps=img2img_steps,
            generator=generator,
        ).images
    t_img2img = t.elapsed

    if "t_load" in performance:
        performance["t_load"] = performance["t_load"] + t_load
    performance["t_preprocess"] = t_preprocess
    performance["t_img2img"] = t_img2img
    performance["t_inpaint"] = performance["t_inference"]
    performance["t_inference"] = sum([performance["t_inference"], t_img2img])

    for i, image in enumerate(final_images):
        image.save("final_image_" + str(i) + ".png")
    return {
        "images": final_images,
        "performance": performance,
    }


if __name__ == "__main__":
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
