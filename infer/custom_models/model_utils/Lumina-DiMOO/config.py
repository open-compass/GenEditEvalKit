# -*- coding: utf-8 -*-
"""
Configuration file
Contains commonly used parameters and settings
"""

# Generation related configuration
GENERATION_CONFIG = {
    "default_timesteps": 64,
    "default_temperature": 1.0,
    "default_cfg_scale": 4.0,
    "default_cfg_img": 4.0,
    "default_seq_len": 1024,
    "default_newline_every": 16,
    "remasking_strategy": "low_confidence"
}

# Image related configuration
IMAGE_CONFIG = {
    "default_height": 512,
    "default_width": 512,
    "max_height": 1024,
    "max_width": 1024,
}

# Special token IDs
SPECIAL_TOKENS = {
    "mask_token": 126336,
    "newline_token": 126084,
    "image_token_offset": 126356,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,  # begin of image
    "eoi": 126350,  # end of image
    "uncondition": 126351
}

# Prompt templates
PROMPT_TEMPLATES = {
    "text_understanding": "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer.",
    "image_generation": "Generate an image according to the text prompt.",
    "image_editing": "Generate an image applying the following editing instruction based on the original image.",
    "dense_prediction": "Perform dense prediction on the given images.",
    "control_generation": "Generate an image according to the text prompt and the given control image.",
    "subject_generation": "Generate an image according to the text prompt and the given object image.",
    "multi_view": "Generate a view-image based on the given image.",
    "style_transfer": "Transform the current image into the style of the provided image."
}

# Edit type configuration
EDIT_TYPE_CONFIG = {
    "dense": {
        "canny": "canny edge map",
        "hed": "hed edge map", 
        "depth": "depth map",
        "openpose": "pose estimation map"
    },
    "supported_types": [
        "canny_pred", "hed_pred", 
        "depth_pred", "openpose_pred", "canny_control", 
        "hed_control", "depth_control", "openpose_control", "subject_driven", 
        "edit", "ref_transfer", "multi_view"
    ]
}
