# Modified from ByteDance-Seed/Bagel/inference.ipynb

import sys
import os
# Add Bagel_repo to sys.path, so that we can directly call the code in Bagel_repo
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{dir_name}/Bagel_repo')

from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from safetensors.torch import load_file

import random
import numpy as np

from .tools import split_input_list

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae

from inferencer import InterleaveInferencer

class BagelEdit:
    def __init__(self, model_path, generate_with_think=False,
                 cfg_text_scale=4.0, cfg_img_scale=2.0,
                 cfg_interval=[0.0, 1.0],
                 timestep_shift=3.0, num_timesteps=50,
                 cfg_renorm_min=0.0, cfg_renorm_type="text_channel",
                 max_think_token_n=1000, do_sample=False):
        self.model_name = model_path.split('/')[-1]
        self.inferencer = self.load_inferencer(model_path)
        self.generate_with_think = generate_with_think

        # Save sampling related hyperparameters
        self.cfg_text_scale = cfg_text_scale
        self.cfg_img_scale = cfg_img_scale
        self.cfg_interval = cfg_interval
        self.timestep_shift = timestep_shift
        self.num_timesteps = num_timesteps
        self.cfg_renorm_min = cfg_renorm_min
        self.cfg_renorm_type = cfg_renorm_type
        self.max_think_token_n = max_think_token_n
        self.do_sample = do_sample

    def load_inferencer(self, model_path):
        # Setting llm_config
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # Setting vit_config
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # Setting vae_model and vae_config
        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        # Integrate into a complete config
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config, 
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model      = SiglipVisionModel(vit_config)
            model          = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Tokenizer Preparing
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Image Transform Preparing
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Multi-GPU inference preparation
        max_mem_per_gpu = "80GiB" # Please adjust according to actual GPU memory
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        print(device_map)
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]
        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
                else:
                    device_map[k] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
        
        # Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        model = model.eval()
        print('Model loaded')

        # Preparing inferencer
        return InterleaveInferencer(
            model=model, 
            vae_model=vae_model, 
            tokenizer=tokenizer, 
            vae_transform=vae_transform, 
            vit_transform=vit_transform, 
            new_token_ids=new_token_ids
        )

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate(self, input_list, seed=42):
        self.set_seed(seed)

        if self.generate_with_think:
            inference_hyper = dict(
                max_think_token_n=self.max_think_token_n,
                do_sample=self.do_sample,
                cfg_text_scale=self.cfg_text_scale,
                cfg_img_scale=self.cfg_img_scale,
                cfg_interval=self.cfg_interval,
                timestep_shift=self.timestep_shift,
                num_timesteps=self.num_timesteps,
                cfg_renorm_min=self.cfg_renorm_min,
                cfg_renorm_type=self.cfg_renorm_type,
            )
        else:
            inference_hyper = dict(
                cfg_text_scale=self.cfg_text_scale,
                cfg_img_scale=self.cfg_img_scale,
                cfg_interval=self.cfg_interval,
                timestep_shift=self.timestep_shift,
                num_timesteps=self.num_timesteps,
                cfg_renorm_min=self.cfg_renorm_min,
                cfg_renorm_type=self.cfg_renorm_type,
            )

        output_list = self.inferencer.interleave_inference(input_lists=input_list, think=self.generate_with_think, **inference_hyper)
        output_dict = {'image': None, 'text': None}
        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i

        # if self.generate_with_think:
        #     print(output_dict['text']) ### for debug

        return output_dict['image']

if __name__ == '__main__':
    # image = Image.open('./Bagel_repo/test_images/women.jpg')
    # prompt = 'She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes.'

    image1 = Image.open('./Bagel_repo/test_images/octupusy.jpg')
    image2 = Image.open('./Bagel_repo/test_images/women.jpg')
    prompt = 'Create a scene where the women in the second image is seeing the billboard in the first image.'
    input_list = [image1, image2, prompt]

    bagel = BagelEdit(model_path='...', generate_with_think=True)
    output_bagel = bagel.generate(input_list=input_list)

    output_bagel.save('test_output/output_bagel_think_edit.png')