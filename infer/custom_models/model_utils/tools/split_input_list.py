from PIL import Image

def split_input_list(input_list, single_image=True):
    # split input_list to images and a combined prompt
    # single_image: whether support multiple images as input. If True, only one input image is allowed.
    input_images = []
    prompts = []
    for item in input_list:
        if isinstance(item, Image.Image):
            input_images.append(item.convert("RGB")) # 防止有些图片不是RGB格式，导致报错
        elif isinstance(item, str):
            prompts.append(item)
        else:
            raise ValueError(f"Unsupported input type {type(item)}. The input list:\n{input_list}")
    prompt = " ".join(prompts)
    if single_image:
        if len(input_images) != 1:
            raise ValueError(f"If and one if one input image is allowed, but got {len(input_images)} images.")
        return input_images[0], prompt
    else:
        return input_images, prompt