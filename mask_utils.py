import numpy as np
import torch

from diffusers.utils import load_image
from PIL import Image, ImageDraw, ImageFilter


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.001] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def create_gradient(image, y=None, x=None, offset=40):
    """
    Takes a binary mask (white = area to be inpainted, black = area to be kept from original image) and creates a gradient at the border of the mask. The gradient adds a white to black gradient that extends into the original black area. 
    
    This ensures that the inpainted area is not a hard border, but a smooth transition from the inpainted area to the original image. 
    
    Used to blend together latents in MaskedImg2ImgPipeline
    """
    if y is None and x is None:
        raise ValueError("Either y or x must be specified")
    draw = ImageDraw.Draw(image)
    if y and x:
        raise ValueError("Only one of y or x must be specified (for now)")

    sign = 1
    if offset < 0:
        sign = -1
    
    offset = abs(offset)
    if y is not None:
        if y > 0: 
            y = image.height - y
            if offset > 0:
                sign = -1 
        else:
            y = abs(y)
        for i in range(abs(offset)):
            color = abs(255 - int(255 * (i / abs(offset))))  # calculate grayscale color
            i *= sign
            draw.line([(0, y+i), (image.width, y+i)], fill=(color, color, color))
    if x is not None:
        if x > 0: 
            x = image.width - x
            if offset > 0:
                sign = -1 
        else:
            x = abs(x)
        for i in range(abs(offset)):
            color = abs(255 - int(255 * (i / abs(offset))))  # calculate grayscale color
            i *= sign
            draw.line([(x+i, 0), (x+i, image.height)], fill=(color, color, color))
    return image

# def soften_mask(mask_before_blur, mask_img, blur_radius):
#     # Apply Gaussian Blur to the mask
#     blurred_mask = mask_img.filter(ImageFilter.GaussianBlur(blur_radius))
#     mask_before_blur = mask_before_blur.convert("L")
    
#     blurred_mask.paste(mask_before_blur, mask=mask_before_blur)

#     return blurred_mask

def expand_image(img, expand_y=0, expand_x=0):
    # Load the image
    img = load_image(img)
    width, height = img.size

    # Create a new image with expanded height
    new_height = height + abs(expand_y)
    new_width = width + abs(expand_x)

    new_img = Image.new('RGB', (new_width, new_height), color = 'white')

    # Create a mask image
    mask_img = Image.new('1', (new_width, new_height), color = 'white')

    # If expand_y is positive, the image is expanded on the bottom.
    # If expand_y is negative, the image is expanded on the top.
    y_position = 0 if expand_y > 0 else abs(expand_y)
    x_position = 0 if expand_x > 0 else abs(expand_x)
    new_img.paste(img, (x_position, y_position))

    # Create mask
    mask_img.paste(Image.new('1', img.size, color = 'black'), (x_position, y_position))
    mask_img = mask_img.convert("RGB")

    # soft_mask_img = soften_mask(mask_img, mask_img, 50)
    # return new_img, mask_img, soft_mask_img
    
    return new_img, mask_img

if __name__ == '__main__':
    # Usage:
    path = "/home/erwann/diffusers/examples/community/castle.png"
    expand = 256
    new_img, mask_img = expand_image(path, expand_x=expand)
    new_img.save('new_image.png')
    mask_img.save('mask_image.png')
    # soft_mask.save('soft_mask.png')
    softened_mask = create_gradient(mask_img, x=expand, offset=40)
    softened_mask.save('soft_mask.png')

