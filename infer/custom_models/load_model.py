import os
file_dir = os.path.dirname(os.path.abspath(__file__))

import importlib

# Unified model configuration:
# First-level key: model_task, such as "t2i" / "edit"
# Second-level key: model_name, such as "bagel", "bagel-think", "qwen-image-edit", etc.
# value: {
#   "model_path":   model path (for local checkpoint) or model hub name (for online models or APIs),
#   "module":       string module path (relative to the current package),
#   "class":        string class name,
#   "model_kwargs": custom parameters (optional, default is {})
# }
MODEL_SETTINGS = {
    # ===================== t2i =====================
    "t2i": {
        "ovis-u1": {
            "model_path": os.path.join(file_dir, "checkpoints/Ovis-U1-3B"),
            "module": ".model_utils.ovis_u1_t2i",
            "class": "OvisU1T2I",
            "model_kwargs": {},
        },
        "bagel": {
            "model_path": os.path.join(file_dir, "checkpoints/BAGEL-7B-MoT"),
            "module": ".model_utils.bagel_t2i",
            "class": "BagelT2I",
            "model_kwargs": {},
        },
        "bagel-think": {
            "model_path": os.path.join(file_dir, "checkpoints/BAGEL-7B-MoT"),
            "module": ".model_utils.bagel_t2i",
            "class": "BagelT2I",
            "model_kwargs": {"generate_with_think": True},
        },
        "lumina-dimoo": {
            "model_path": os.path.join(file_dir, "checkpoints/Lumina-DiMOO"),
            "module": ".model_utils.lumina_dimoo_t2i",
            "class": "LuminaDiMOOT2I",
            "model_kwargs": {},
        },
        "qwen-image": {
            "model_path": os.path.join(file_dir, "checkpoints/Qwen-Image"),
            "module": ".model_utils.qwen_image_t2i",
            "class": "QwenImageT2I",
            "model_kwargs": {},
        },
        "qwen-image-2512": {
            "model_path": os.path.join(file_dir, "checkpoints/Qwen-Image-2512"),
            "module": ".model_utils.qwen_image_t2i",
            "class": "QwenImageT2I",
            "model_kwargs": {},
        },
    },

    # ===================== edit =====================
    "edit": {
        "ovis-u1": {
            "model_path": os.path.join(file_dir, "checkpoints/Ovis-U1-3B"),
            "module": ".model_utils.ovis_u1_edit",
            "class": "OvisU1Edit",
            "model_kwargs": {},
        },
        "bagel": {
            "model_path": os.path.join(file_dir, "checkpoints/BAGEL-7B-MoT"),
            "module": ".model_utils.bagel_edit",
            "class": "BagelEdit",
            "model_kwargs": {},
        },
        "bagel-think": {
            "model_path": os.path.join(file_dir, "checkpoints/BAGEL-7B-MoT"),
            "module": ".model_utils.bagel_edit",
            "class": "BagelEdit",
            "model_kwargs": {"generate_with_think": True},
        },
        "lumina-dimoo": {
            "model_path": os.path.join(file_dir, "checkpoints/Lumina-DiMOO"),
            "module": ".model_utils.lumina_dimoo_edit",
            "class": "LuminaDiMOOEdit",
            "model_kwargs": {},
        },
        # Support both "qwen-image" and "qwen-image-edit" as names for the same model
        "qwen-image": {
            "model_path": os.path.join(file_dir, "checkpoints/Qwen-Image-Edit"),
            "module": ".model_utils.qwen_image_edit",
            "class": "QwenImageEdit",
            "model_kwargs": {},
        },
        "qwen-image-edit": {
            "model_path": os.path.join(file_dir, "checkpoints/Qwen-Image-Edit"),
            "module": ".model_utils.qwen_image_edit",
            "class": "QwenImageEdit",
            "model_kwargs": {},
        },
        "qwen-image-edit-2509": {
            "model_path": os.path.join(file_dir, "checkpoints/Qwen-Image-Edit-2509"),
            "module": ".model_utils.qwen_image_edit_plus",
            "class": "QwenImageEditPlus",
            "model_kwargs": {},
        },
        "qwen-image-edit-2511": {
            "model_path": os.path.join(file_dir, "checkpoints/Qwen-Image-Edit-2511"),
            "module": ".model_utils.qwen_image_edit_plus",
            "class": "QwenImageEditPlus",
            "model_kwargs": {},
        },
    },
}


def load_model(model_name, model_task, gpu_id=None, custom_model_kwargs=None):
    """
    Load model based on task and model name.
    - model_task: "t2i" or "edit"
    - model_name: e.g., "bagel", "bagel-think", "qwen-image", "qwen-image-edit", etc.
    - custom_model_kwargs: Additional parameters passed to the model class (optional). This should be a dictionary that overrides the default model_kwargs.
    """
    if gpu_id is not None: # Load the model onto a specified GPU for multi-process scenarios
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Load model
    if model_task not in MODEL_SETTINGS:
        raise ValueError(f"Unsupported model task: {model_task}")
    
    supported_models = MODEL_SETTINGS[model_task]

    # Check if the model exists for the given task
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model name '{model_name}' for task '{model_task}'")

    setting = supported_models[model_name]

    model_path = setting["model_path"]
    module_name = setting["module"]
    class_name = setting["class"]
    model_kwargs = dict(setting.get("model_kwargs", {}))

    if custom_model_kwargs:
        model_kwargs.update(custom_model_kwargs)

    # Import as needed: only import the corresponding module when this model is needed to avoid errors when the environment is incompatible
    module = importlib.import_module(module_name, package=__package__)
    cls = getattr(module, class_name)

    model = cls(model_path=model_path, **model_kwargs)
    return model