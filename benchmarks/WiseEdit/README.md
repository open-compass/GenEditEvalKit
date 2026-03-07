<h1 align="center" style="line-height: 50px;">
  WiseEdit: Benchmarking Cognition- and Creativity-Informed Image Editing
</h1>

<div align="center">

Kaihang Pan<sup>1</sup>\* · Weile Chen<sup>1</sup>\* · Haiyi Qiu<sup>1</sup>\* · Qifan Yu<sup>1</sup> · Wendong Bu<sup>1</sup> · Zehan Wang<sup>1</sup> ·  
Yun Zhu<sup>2</sup> · Juncheng Li<sup>1</sup> · Siliang Tang<sup>1</sup>  

<sup>1</sup>Zhejiang University &nbsp;&nbsp;&nbsp; <sup>2</sup>Shanghai Artificial Intelligence Laboratory  

\*Equal contribution.

[![arXiv](https://img.shields.io/badge/arXiv-2512.00387-b31b1b.svg)](https://www.arxiv.org/abs/2512.00387)
[![Project Page](https://img.shields.io/badge/Home-Page-b3.svg)](https://qnancy.github.io/wiseedit_project_page/)
[![Dataset](https://img.shields.io/badge/🤗%20Huggingface-WiseEdit_dataset-yellow)](https://huggingface.co/datasets/123123chen/WiseEdit-Benchmark)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github)](https://github.com/beepkh/WiseEdit)

</div>

---

# 🌍 Introduction

WiseEdit is a knowledge-intensive benchmark for cognition- and creativity-informed image editing. It decomposes instruction-based editing into three stages, **Awareness**, **Interpretation**, and **Imagination**, and provides **1,220 bilingual test cases** together with a GPT-4o–based automatic evaluation pipeline. Using WiseEdit, we benchmark **22 state-of-the-art image editing models** and reveal clear limitations in knowledge-based reasoning and compositional creativity.  

<p align="center">
  <img src="figures/intro.png" width="100%">
</p>

# 🔥 News

- **[2025.11.29]** 📄 WiseEdit paper released on arXiv.  
- **[2025.11.29]** 📊 WiseEdit project page released.
- More updates coming soon – stay tuned and ⭐ star the repo!

### TODO

- [x] Release paper and project page.  
- [x] Release WiseEdit benchmark data.  
- [x] Release automatic evaluation code & prompts.  
- [x] Release baseline results & model outputs.  

# 💡 Overview

WiseEdit is built around **task depth** and **knowledge breadth**.
<p align="center">
  <img src="figures/wiseedit-intro.png" width="90%">
</p>

## Task Depth – Four Task Types

WiseEdit includes: 

- **Awareness Task**  
  - Focus: *Where* to edit.  
  - No explicit spatial coordinates are given in the instruction.  
  - Requires comparative reasoning, reference matching, or fine-grained perception  
- **Interpretation Task**  
  - Focus: *How* to edit at the perception level.  
  - Instructions often encode **implicit intent**, demanding world knowledge  
- **Imagination Task**  
  - Focus: subject driven creative generation.  
  - Requires complex composition and identity-preserving transformations  

- **WiseEdit-Complex**  
  - Combines Awareness + Interpretation + Imagination.  
  - Multi-image, multi-step reasoning with conditional logic and compositional generation.

## Knowledge Breadth – Three Knowledge Types

WiseEdit organizes cases by **knowledge type**: 

- **Declarative Knowledge** – “knowing what”  
  - Facts, concepts, perceptual cues.  

- **Procedural Knowledge** – “knowing how”  
  - Multi-step skills or procedures.  

- **Metacognitive Knowledge** – “knowing about knowing”  
  - When and how to apply declarative / procedural knowledge; conditional reasoning, rule stacking, etc.  

These are grounded in **Cultural Common Sense**, **Natural Sciences**, and **Spatio-Temporal Logic**, stressing culturally appropriate, physically consistent, and logically coherent edits.

# ⭐ Evaluation Protocol

We adopt a **VLM-based automatic evaluation pipeline**:

- **Backbone evaluator**: GPT-4o.  
- **Metrics (1–10 → linearly mapped to 0–100)**:   
  - **IF** – Instruction Following  
  - **DP** – Detail Preserving  
  - **VQ** – Visual Quality  
  - **KF** – Knowledge Fidelity (for knowledge-informed cases)  
  - **CF** – Creative Fusion (for imagination / complex cases)  

The **overall score** is:
$\text{AVG} = \frac{\text{IF} + \text{DP} + \text{VQ} + \alpha \cdot \text{KF} + \beta \cdot \text{CF}}{3 + \alpha + \beta}$

where $\alpha$ and $\beta$ are 1 only when KF / CF are applicable.
Our user study shows strong correlation between this protocol and human ratings.  

# 📊 Dataset & Results
### WiseEdit-Benchmark
Our benchmark data is hosted on Hugging Face:  
- **WiseEdit-Benchmark**: https://huggingface.co/datasets/123123chen/WiseEdit-Benchmark  

The folder structure for WiseEdit-Benchmark is organized as follows:
```text
WiseEdit-Benchmark/
├── WiseEdit/
│   ├── Awareness/
│   │   ├── Awareness_1/
│   │   │   ├── imgs/                  # input images for this subset
│   │   │   ├── img_ref/               # reference images (if any)
│   │   │   ├── Awareness_1.csv        # metadata + instructions in CSV format
│   │   │   └── ins.json               # same annotations in JSON format (used by code)
│   │   └── Awareness_2/
│   │       ├── imgs/
│   │       ├── img_ref/
│   │       ├── Awareness_2.csv        # metadata + instructions in CSV format
│   │       └── ins.json               # same annotations in JSON format
│   ├── Imagination/
│   │   └── ...                        # similar structure for Imagination subsets
│   └── Interpretation/
│       └── ...                        # similar structure for Interpretation subsets
└── WiseEdit-Complex/
    ├── WiseEdit_Complex_2/
    │   ├── imgs/
    │   ├── img_ref/
    │   ├── WiseEdit_Complex_2.csv     # metadata + instructions in CSV format
    │   └── ins.json                   # same annotations in JSON format
    ├── WiseEdit_Complex_3/
    │   └── ...
    └── WiseEdit_Complex_4/
        └── ...
 ```

### WiseEdit-Results
All our model evaluation results are also released at:  
- **WiseEdit-Results**: https://huggingface.co/datasets/midbee/WiseEdit-Results

# 🚀 Usage
## Environment setup
This project requires Python 3.10.
First install dependencies (from requirements.txt):

`pip install -r requirements.txt`

Set your API credentials (the evaluator calls an OpenAI-compatible Chat Completions API):

```
# required
export API_KEY="YOUR_API_KEY"
# optional: if not set, the default https://api.openai.com/v1 will be used
export BASE_URL="https://api.openai.com/v1"
```
If BASE_URL is not set, it will automatically fall back to https://api.openai.com/v1.

### Example with conda start
```
git clone https://github.com/beepkh/WiseEdit
cd WiseEdit

# 1) create and activate env
conda create -n wiseedit python=3.10 -y
conda activate wiseedit

# 2) install requirements
pip install -r requirements.txt

# 3) set env vars
export API_KEY="YOUR_API_KEY"
# optional
export BASE_URL="https://api.openai.com/v1"
```
## Step 1: Organizing Generated Images
Before running the evaluation, you need to organize all **generated images** as:
`result_img_root/<MODEL_NAME>/<SUBSET>/<LANG>/<ID>.png`,
where `<MODEL_NAME>` is the model tag passed to `--name`, `<SUBSET>` is the CSV/JSON subset name (e.g. `Awareness_1`, `Imagination_2`, `WiseEdit_Complex_3`), `<LANG>` is `cn` or `en`, and `<ID>.png` is the sample id in the corresponding CSV/JSON (e.g. `1.png`, `2.png`, …).

You can refer to the [WiseEdit-Results](https://huggingface.co/datasets/midbee/WiseEdit-Results) for an example of this directory layout.
```text
/path/to/result_images_root/
└── <MODEL_NAME>/                 # e.g. Nano-banana-pro, GPT, etc.
    ├── Awareness_1/
    │   ├── cn/
    │   │   ├── 1.png             # id = 1 in Awareness_1.csv / ins.json (cn)
    │   │   ├── 2.png
    │   │   └── ...
    │   └── en/
    │       ├── 1.png             # id = 1 in Awareness_1.csv / ins.json (en)
    │       ├── 2.png
    │       └── ...
    ├── Awareness_2/
    │   ├── cn/
    │   └── en/
    ├── Imagination_1/
    │   ├── cn/
    │   └── en/
    ├── Imagination_2/
    │   └── ...
    ├── Interpretation_1/
    │   └── ...
    ├── WiseEdit_Complex_2/
    │   └── ...
    └── ...
```

`Evaluation/generate_image_example.py` uses FLUX.2-Dev as an example to demonstrate how to generate the corresponding images for each test case in WiseEdit.


## Step 2: Run evaluation
Run `run_eval.py` to score all subsets and produce `score_*.csv`:

```
python run_eval.py \
  --name Nano-banana-pro \
  --dataset_dir /path/to/WiseEdit-Benchmark \
  --result_img_root /path/to/result_images_root \
  --score_output_root /path/to/score_output_root \
  --num_workers 5 # number of threads used for evaluation
```

To evaluate only specific CSVs (e.g. Imagination_1.csv and Awareness_1.csv):

```
python run_eval.py \
  --name Nano-banana-pro \
  --dataset_dir /path/to/WiseEdit-Benchmark \
  --result_img_root /path/to/result_images_root \
  --score_output_root /path/to/score_output_root \
  --num_workers 5 \
  --target_csv Imagination_1.csv Awareness_1.csv
```

`run_eval.py` will write files like:

```
/score_output_root/Nano-banana-pro/score_Imagination_1.csv
/score_output_root/Nano-banana-pro/score_Awareness_1.csv
...
```

## Step 3: Aggregate statistics

After all `score_*.csv` are ready, run `statistic.py`:
```
python statistic.py \
  --dataset_dir /path/to/WiseEdit-Benchmark \
  --score_root /path/to/score_output_root \
  --name Nano-banana-pro \
  --statistic_output_dir /path/to/statistic_output 
```

This will generate:
```
/statistic_output/Nano-banana-pro_cn.csv
/statistic_output/Nano-banana-pro_en.csv
/statistic_output/Nano-banana-pro_complex.csv
```

and print per-task, per-language averages to the console (You can replace Nano-banana-pro with your own model-name).

**If you only want to test the results under single image (like Table 5 and Table 6 in our paper)**, After all `score_*_1.csv` are ready (note there is no score_WiseEdit_Complex_1.csv), run `statistic_single.py`:
```
python statistic_single.py \
  --dataset_dir /path/to/WiseEdit-Benchmark \
  --score_root /path/to/score_output_root \
  --name Nano-banana-pro \
  --statistic_output_dir /path/to/statistic_output 
```

This will generate:
```
/statistic_output/Nano-banana-pro_cn_sing.csv
/statistic_output/Nano-banana-pro_en_sing.csv
```
and print per-task, per-language averages to the console.

# ✍️Citation

If you find WiseEdit helpful, please cite:
```bibtex
@article{pan2025wiseedit,
  title={WiseEdit: Benchmarking Cognition-and Creativity-Informed Image Editing},
  author={Pan, Kaihang and Chen, Weile and Qiu, Haiyi and Yu, Qifan and Bu, Wendong and Wang, Zehan and Zhu, Yun and Li, Juncheng and Tang, Siliang},
  journal={arXiv preprint arXiv:2512.00387},
  year={2025}
}
```



