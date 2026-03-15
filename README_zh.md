# 项目简介

在图像生成和编辑模型的评测中，如何**高效地同时评测多个模型与多个 benchmark** 一直是一个难点。传统做法往往需要为每个模型与每个 benchmark 单独编写脚本，流程分散、维护成本高。

为了解决这一问题，我们开发了 **GenEditEvalKit** —— 一个通用的图像生成和编辑模型评测工具包。它提供统一的评测入口与配置方式，支持多模型、多 benchmark 的并行评测，并提供简单易扩展的接口，方便你添加新的模型和 benchmark。该工具包也可用于评测理解生成一体化模型（UMMs）。

[⚡ 快速开始](#快速开始)
 
 ---
 
# 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
  - [环境配置](#环境配置)
  - [启动命令](#启动命令)
    - [一、配置参数说明](#一配置参数说明)
    - [二、配置优先级说明](#二配置优先级说明)
    - [三、启动方式](#三启动方式)
- [开发指南](#开发指南)
  - [部署新模型](#部署新模型)
  - [部署新 benchmark](#部署新-benchmark)
- [项目结构](#项目结构)
- [对于 GPU 无法联网的场景](#对于-gpu-无法联网的场景)

---
 
# 功能特性

1. **多模型 × 多 benchmark 统一评测**  
   - 相比传统脚本通常只能评测「单模型 × 单 benchmark」的组合，GenEditEvalKit 支持在一次运行中同时评测多个模型在多个 benchmark 上的表现；  
   - 无需为每个组合单独写脚本，在不改代码的前提下即可灵活切换和组合模型 / benchmark，大幅降低评测脚本的维护成本、提升整体评测效率。

2. **并行评测与资源利用**  
   - 支持并行运行多个评测任务，充分利用 GPU / CPU 等计算资源，显著缩短整体评测时间；  
   - 并行度可以按硬件条件灵活配置，以在「吞吐量」与「单次评测显存占用」之间做权衡。

3. **易扩展的模块化与注册式框架**  
   - 模型与 benchmark 均采用模块化 + 注册式管理，方便在现有框架上快速扩展；  
   - 集成新模型时，只需按照统一接口实现模型调用逻辑，并在指定位置完成注册，即可纳入统一推理流程；  
   - 集成新 benchmark 时，只需实现对应的数据集类和评测脚本并完成注册，无需改动核心调度逻辑，即可接入完整的「推理 + 评测」流水线。

4. **统一的结果输出与日志管理**  
   - 对不同 benchmark 的模型评测结果和日志采用统一的目录结构组织；  
   - 支持按「模型 × benchmark」快速定位推理输出与评测指标，方便后续对比分析、可视化和问题排查。

5. **已集成多个生成和编辑的 Benchmark**
   - 目前已集成 **12 个生成 benchmark** 和 **4 个编辑 benchmark**，下载仓库后可直接使用
   - 编辑 benchmark 通常包含输入图片，导致体积较大。为控制仓库大小，仅保留体积较小的 imgedit 作为参考实现。用户可基于此自行部署其他编辑 benchmark
   - 后续我们将持续补充新 benchmark，也欢迎社区贡献（PR）

## 当前支持的 benchmark
<table>
  <thead>
    <tr>
      <th align="left">T2I Benchmark</th>
      <th align="left">Original Repository</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>dpgbench</code></td><td><a href="https://github.com/TencentQQGYLab/ELLA">TencentQQGYLab/ELLA</a></td></tr>
    <tr><td><code>genai</code></td><td><a href="https://huggingface.co/datasets/BaiqiL/GenAI-Bench">BaiqiL/GenAI-Bench</a></td></tr>
    <tr><td><code>geneval</code></td><td><a href="https://github.com/djghosh13/geneval">djghosh13/geneval</a></td></tr>
    <tr><td><code>geneval2</code></td><td><a href="https://github.com/facebookresearch/GenEval2">facebookresearch/GenEval2</a></td></tr>
    <tr><td><code>genexam</code></td><td><a href="https://github.com/OpenGVLab/GenExam">OpenGVLab/GenExam</a></td></tr>
    <tr><td><code>hpsv2</code></td><td><a href="https://github.com/tgxs002/HPSv2">tgxs002/HPSv2</a></td></tr>
    <tr><td><code>longtext</code></td><td><a href="https://github.com/X-Omni-Team/X-Omni">X-Omni-Team/X-Omni</a></td></tr>
    <tr><td><code>oneig</code></td><td><a href="https://github.com/OneIG-Bench/OneIG-Benchmark">OneIG-Bench/OneIG-Benchmark</a></td></tr>
    <tr><td><code>t2ireasonbench</code></td><td><a href="https://github.com/KaiyueSun98/T2I-ReasonBench">KaiyueSun98/T2I-ReasonBench</a></td></tr>
    <tr><td><code>tiff</code></td><td><a href="https://github.com/A113N-W3I/TIIF-Bench">A113N-W3I/TIIF-Bench</a></td></tr>
    <tr><td><code>unigenbench</code></td><td><a href="https://github.com/CodeGoat24/UniGenBench">CodeGoat24/UniGenBench</a></td></tr>
    <tr><td><code>wise</code></td><td><a href="https://github.com/PKU-YuanGroup/WISE">PKU-YuanGroup/WISE</a></td></tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th align="left">Editing Benchmark</th>
      <th align="left">Original Repository</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>gedit/gedit_cn</code></td><td><a href="https://github.com/stepfun-ai/Step1X-Edit">stepfun-ai/Step1X-Edit</a></td></tr>
    <tr><td><code>imgedit</code></td><td><a href="https://github.com/PKU-YuanGroup/ImgEdit">PKU-YuanGroup/ImgEdit</a></td></tr>
    <tr><td><code>krisbench</code></td><td><a href="https://github.com/mercurystraw/Kris_Bench">mercurystraw/Kris_Bench</a></td></tr>
    <tr><td><code>risebench</code></td><td><a href="https://github.com/PhoenixZ810/RISEBench">PhoenixZ810/RISEBench</a></td></tr>
    <tr><td><code>wiseedit</code></td><td><a href="https://github.com/beepkh/WiseEdit">beepkh/WiseEdit</a></td></tr>
  </tbody>
</table>

---
 
# 快速开始

下面给出一个**可直接运行**的示例：通过 **OpenAI API 兼容接口**直接评测 `GPT-Image-1.5` 在 `Imgedit` 上的表现

## 运行前提
在执行命令前，请确保在 `config.sh` 中完成以下配置：
1. **API 配置**：填写 `API_KEY`（如使用自定义网关，还需填写 `BASE_URL`）。
2. **路径配置**：填写 `CONDA_BASE` 为你的 conda 安装路径。
3. **环境映射**：在 `INFER_ENV_MAP` 中为 `gpt-image-1.5` 指定环境（需安装 `openai` 库和 `Pillow`库），在 `EVAL_ENV_MAP` 中为 `imgedit` 指定环境（需通过 `pip install -r requirements/benchmark/edit/imgedit.txt` 配置）。

## 执行命令
```bash
bash eval.sh --model_names "gpt-image-1.5" --ds_names "imgedit" --use_api true --num_workers 4 --infer true --eval true
```

---
 
# 使用指南

## 环境配置

请根据你**计划评测的模型和 benchmark**，在 `requirements/` 目录下找到对应的环境依赖文件，并创建相应的环境。例如，假设你需要评测的模型是 `Bagel`，benchmark 是 `GenEval`，则可以执行：

```bash
pip install -r requirements/model/bagel.txt
pip install -r requirements/benchmark/t2i/geneval.txt
```

注：上述 requirements.txt 我们已尽量与原始模型或 benchmark 仓库中的依赖文件保持一致，但原始文件本身可能存在问题。若出现安装或运行报错，请根据具体情况灵活调整依赖或版本。

---

## 启动命令

在运行主评测脚本 `eval.sh` 前，**推荐先在 `config.sh` 中完成参数配置**。  
此外，部分参数除了在 `config.sh` 中配置外，也还可以通过命令行参数临时覆盖。具体可以参照下文的说明。

---

### 一、配置参数说明

下表为 `config.sh` 中的主要参数及其含义，并标明是否支持被命令行覆盖：

| 参数名 | 是否支持命令行覆盖 | 说明 | 示例 |
|--------|--------------------|------|------|
| `DS_NAMES` | ✅ 支持 | 需要评测的 benchmark 名称列表。多个 benchmark 使用空格分隔。若通过命令行传入 `--ds_names`，则命令行配置会覆盖此处设置。 | `DS_NAMES=('genai' 'wiseedit')` |
| `MODEL_NAMES` | ✅ 支持 | 需要评测的模型名称列表。多个模型使用空格分隔。若通过命令行传入 `--model_names`，则命令行配置会覆盖此处设置。 | `MODEL_NAMES=('bagel' 'qwen-image')` |
| `CUSTOM_MODEL_KWARGSES`| ✅ 支持 | 为每个模型传递自定义参数，用分号分隔。| `--custom_model_kwargses "p1=v1;p2=v2"` |
| `USE_API` | ✅ 支持 | 推理阶段是否使用 API。 | `--use_api true` |
| `NUM_WORKERS` | ✅ 支持 | API 推理的并发数，当且仅当 `USE_API=true` 时生效。 | `--num_workers 4` |
| `PARALLEL` | ✅ 支持 | 是否开启多模型 × 多 benchmark 组合的**并行评测**。`true` 可提升整体吞吐，但要求显存较为充足。可被命令行参数 `--parallel` 覆盖。 | `--parallel true` |
| `ENABLE_INFER` | ✅ 支持 | 是否执行推理阶段（生成图片）。若输出目录下已存在推理结果，可设为 `false`，仅执行评测阶段。可被命令行参数 `--infer` 覆盖。 | `ENABLE_INFER=true` |
| `ENABLE_EVAL` | ✅ 支持 | 是否执行评测阶段（计算指标）。可被命令行参数 `--eval` 覆盖。 | `ENABLE_EVAL=true` |
| `API_KEY` | ❌ 不支持 | 评测时调用部分需要 API 的 benchmark 所用的密钥。当前只能在 `config.sh` 中设置，不支持命令行覆盖。 | `API_KEY="your_api_key_here"` |
| `BASE_URL` | ❌ 不支持 | 调用评测所依赖 API 的 base url，同样只能在 `config.sh` 中进行配置。 | `BASE_URL="https://api.openai.com/v1"` |
| `CONDA_BASE` | ❌ 不支持 | conda 安装路径，用于在各评测脚本中定位到对应环境的 `python`。 | `CONDA_BASE="/path/to/miniconda3"` |
| `CUDA_BASE` | ❌ 不支持 | CUDA 安装根路径，用于部分脚本中手动指定 CUDA 版本。 | `CUDA_BASE="/path/to/cuda"` |
| `TORCH_HOME` | ❌ 不支持 | PyTorch 模型缓存目录。 | `TORCH_HOME="/path/to/.cache/torch"` |
| `HF_HOME` | ❌ 不支持 | HuggingFace 总缓存目录，评测相关模型会缓存到此处。 | `HF_HOME="/path/to/GenEditEvalKit/.cache/huggingface"` |
| `TRANSFORMERS_CACHE` | ❌ 不支持 | Transformers 缓存目录，通常设为 `${HF_HOME}/hub`。 | `TRANSFORMERS_CACHE="$HF_HOME/hub"` |
| `HF_HUB_OFFLINE` | ❌ 不支持 | 是否启用 HuggingFace 离线模式。`1` 表示**完全离线**（不再联网下载），`0` 表示允许联网下载。 | `HF_HUB_OFFLINE=1` |
| `TRANSFORMERS_OFFLINE` | ❌ 不支持 | 是否启用 Transformers 离线模式，语义同 `HF_HUB_OFFLINE`。 | `TRANSFORMERS_OFFLINE=1` |
| `INFER_ENV_MAP` | ❌ 不支持 | **推理阶段**模型名到 conda 环境名的映射。key 为模型名称，value 为该模型使用的推理环境名。目前只能在 `config.sh` 中配置。 | `INFER_ENV_MAP=(['bagel']='bagel-env' ['qwen-image']='qwen-image-env')` |
| `EVAL_ENV_MAP` | ❌ 不支持 | **评测阶段** benchmark 名称到 conda 环境名的映射。部分 benchmark 需要设置多个环境，此时建议以空格分隔。 | `EVAL_ENV_MAP=(['genai']='yph-genai' ['wiseedit']='yph-risebench')` |
| `EVAL_GPU_MAP` | ❌ 不支持 | 各 benchmark 在 **评测阶段** 需要使用的 GPU 数量。key 为 benchmark 名称，value 为 GPU 数；`0` 表示不占用 GPU（如纯 API 评测）。主要用于提前告知调度系统（如 `srun` 等）所需的 GPU 数。 | `EVAL_GPU_MAP=(['genai']=1 ['gedit']=0 ['cvtg']=1)` |
| `GENERATE_BLANK_IMAGE_ON_ERROR` | ❌ 不支持 | 推理阶段若出现异常，是否自动生成**空白图片占位**，以避免单条数据失败导致整轮评测中断。仅在 `config.sh` 中配置。 | `GENERATE_BLANK_IMAGE_ON_ERROR=false` |

> 说明：  
> - 「支持命令行覆盖」表示：若在命令行中传入相应参数（如 `--ds_names genexam` / `--model_names intenvl-u` 等），则**优先使用命令行配置**，忽略 `config.sh` 中对应项；  
> - 其余参数均需通过直接修改 `config.sh` 的方式来修改。

---

### 二、配置优先级说明

整体配置优先级如下：

1. `config.sh`：  
   - 提供**默认配置**，适合相对固定、日常使用的评测场景；
   - 建议将常用的模型 / benchmark / 环境映射等长期设置放在这里。

2. 命令行参数：  
   - 用于临时修改评测范围（如只想跑某几个模型或某几个 benchmark）；  
   - 仅对支持覆盖的参数生效：`ds_names`、`model_names`、`infer`、`eval`。

当同一个参数在两处都设置时，**命令行参数优先级更高**，会覆盖 `config.sh` 中对应项。

示例：

- `config.sh` 中：  
  ```bash
  DS_NAMES=('genai' 'wiseedit')
  ```
- 命令行中：  
  ```bash
  --ds_names "wiseedit;gedit"
  ```

则本次运行实际评测的 benchmark 为：`wiseedit` 与 `gedit`。

---

### 三、启动方式

#### 1. 直接使用 `config.sh` 中的配置

在完成 `config.sh` 的基础配置后，可以直接运行主评测脚本：

```bash
bash eval.sh
```

此时：

- 实际评测的**模型列表**由 `MODEL_NAMES` 决定；
- 实际评测的 **benchmark 列表** 由 `DS_NAMES` 决定；
- 是否执行推理 / 评测分别由 `ENABLE_INFER` 与 `ENABLE_EVAL` 控制；
- 推理 / 评测使用的具体环境与 GPU 数量则由 `INFER_ENV_MAP`、`EVAL_ENV_MAP`、`EVAL_GPU_MAP` 等配置决定。

适用于日常固定评测场景，减少频繁敲命令行参数的成本。

---

#### 2. 通过命令行临时覆盖部分配置

如果你不想修改 `config.sh`，而只是在当前一次运行中临时改变评测范围或开关，可以在命令行中传入参数进行覆盖：

```bash
bash eval.sh \
  --ds_names "wiseedit;gedit" \
  --model_names "bagel;qwen-image" \
  --custom_model_kwargses ";" \
  --infer true \
  --eval true \
```

上述命令的含义：

- 本次只评测 `bagel` 与 `qwen-image` 两个模型；
- 评测的 benchmark 为 `wiseedit` 与 `gedit`；
- 无论 `config.sh` 中默认如何配置，本次运行都会同时执行推理（`infer=true`）和评测（`eval=true`）。

---

# 开发指南

本节介绍如何在 GenEditEvalKit 中**接入新模型**与**接入新 benchmark**。

## 部署新模型

### Step 1. 准备模型权重

建议将模型权重统一下载到：

```text
infer/custom_models/checkpoints/
```

这样可以方便模型的调用。当然，你也可以将模型权重放在其他路径，但需要在后续注册模型加载逻辑时，正确指定权重路径。

---

### Step 2. 实现模型调用接口

在 `infer/custom_models/model_utils` 目录下，为你的模型新建一个接口文件，并实现一个模型类。  
该类需要满足以下要求：

- **必须包含属性：**  
  - `self.model_name`：字符串，用于标识模型名称（需与配置中的 `MODEL_NAMES` 一致或可映射）。

- **必须实现方法：**  
  - `self.generate()`：模型的核心推理函数。

根据模型任务类型不同，`generate` 接口定义如下：

- **文本生成图像模型（Text-to-Image, t2i）**

  ```python
  def generate(self, prompt, seed=42):
      """
      prompt: 文生图prompt
      seed:   inference的随机数种子
      """
      ...
  ```

- **图像编辑模型（Image Editing, edit）**

  ```python
  def generate(self, input_list, seed=42):
      """
      input_list: 图片编辑所需的输入列表，通常为 (str, PIL.Image) 的 list，
                  包含编辑说明及待编辑的图片等信息。
      seed:       inference的随机数种子
      """
      ...
  ```

---

### Step 3. 注册模型加载逻辑

完成模型类定义后，需要在

```text
GenEditEvalKit/infer/custom_models/load_model.py
```

中的 `MODEL_SETTINGS` 变量下注册该模型，以便评测主流程能够根据模型名自动加载。

常用配置字段说明如下：

| 字段名 | 说明 |
|------|------|
| `model_path` | 模型权重存放路径（通常位于 `infer/custom_models/checkpoints/` 下） |
| `module` | 模型调用接口所在的模块路径（Python import 路径） |
| `class` | 模型类名（即你在 `model_utils` 下相应.py脚本中定义的类名） |
| `model_kwargs` | 初始化模型时需要额外传入的参数（`dict`）。若后续在 `config.sh` 或命令行中提供了同名自定义参数，将覆盖这里的默认值。 |

完成上述步骤后，即可在 `config.sh` 的 `MODEL_NAMES` 或命令行的 `--model_names` 中使用该模型名称参与评测。

---

## 部署新 benchmark

### Step 1. 定义数据集类

在下面的目录中，根据任务类型（t2i / edit）创建数据集类：

```text
infer/custom_datasets/dataset_cls/edit/
infer/custom_datasets/dataset_cls/t2i/
```

实现数据集类时，需要满足以下要求：

- **必须包含属性：**  
  - `self.dataset_name`：用于标识 benchmark 名称；  
  - `self.dataset_task`：用于标识该 benchmark 的任务类型（如 `"t2i"` 或 `"edit"` 等）；  
  - `self.data`：benchmark 评测样本的列表。

其中，`self.data` 是一个由 `dict` 组成的 `list`，每个元素表示一条评测指令，结构要求如下：

- **t2i 任务：**

  ```python
  {
      "prompt": ... ,          # 文生图指令
      "seed": ... ,            # 随机种子。若模型评测需要基于同一 prompt 生成多张图片，可通过设置不同 seed 来实现图片生成的多样性
      "image_save_path": ... , # 输出图片的保存路径
  }
  ```

- **edit 任务：**

  ```python
  {
      "input_list": ... ,      # 编辑图片 + 编辑指令的组合列表，一般为 list(str, PIL.Image)
      "seed": ... ,            # 随机种子。若模型评测需要基于同一输入生成多张图片，可通过设置不同 seed 来实现图片编辑的多样性
      "image_save_path": ... , # 输出图片的保存路径
  }
  ```

数据集类准备好之后，请在：

```text
GenEditEvalKit/infer/custom_datasets/load_dataset.py
```

中完成该 benchmark 的注册，使主流程可以通过 benchmark 名称自动加载对应数据集。

---

### Step 2. 编写评测脚本

根据对应 benchmark 官方给出的标准评测命令，编写一个评测脚本，并将其放在：

```text
GenEditEvalKit/eval/
```

目录下。

评测脚本的**输入参数**应至少包括：

- `MODEL_NAME`：当前需要评测的模型名称；  
- `EVAL_ENV`：运行该 benchmark 评测所需的 conda 环境名称（通常由 `EVAL_ENV_MAP` 映射得到）。

### Step 3. 设置评测相关参数
在 `config.sh` 中的 `EVAL_ENV_MAP` 变量下，为该 benchmark 设置对应的 conda 环境名称。
在 `config.sh` 中的 `EVAL_GPU_MAP` 变量下，为该 benchmark 设置评测时所需的 GPU 数量。


> 重要：  
> - 在**训练模型**时，工作目录通常为仓库根目录（例如 `GenEditEvalKit`）；  
> - 在**评测时**，工作目录会切换到对应的 benchmark 仓库目录。  
>   在编写评测脚本时，请务必注意路径的使用。

---

# 项目结构

项目的主要目录结构及功能说明如下：

```text
├── benchmarks/      # 各个 benchmark 的评测仓库
├── eval/            # 评测脚本入口（按 benchmark 划分）
├── eval_models/     # 各 benchmark 评测阶段所需的模型权重（例如clip模型）
├── infer/           # inference 相关代码（统一的推理接口层）
│   ├── custom_datasets/ # benchmark 数据调用接口
│   └── custom_models/
│       ├── checkpoints/ # 各模型权重文件
│       └── model_utils/ # 模型调用接口
├── log/             # 推理与评测阶段的日志（可用于排错与结果回溯）
├── output/          # 保存推理输出的图片与评测结果（按模型和 benchmark 组织）
├── requirements/    # 各模型推理和各 benchmark 评测所需的环境依赖
└── utils/           # 通用辅助工具脚本
    ├── check_empty_image.py      # 检查生成图片是否为空白图片，可用于排除图像生成或编辑的异常
    ├── download_model_hf_home.sh # 手动下载 HuggingFace 模型到 HF_HOME
    └── use_cuda.sh               # 切换 CUDA 版本
```

---

# 对于 GPU 无法联网的场景

部分 benchmark 在评测时会先从 HuggingFace 下载模型，然后才在本地 GPU 上完成部署和推理。如果**GPU 节点无法直接访问公网**，则需要提前在可联网环境中下载所需权重，再拷贝到评测环境中使用。

在本项目中，你可以通过以下方式，将模型手动下载到 `${HF_HOME}/hub` 中（`HF_HOME` 在 `eval.sh` 中配置）：

- 使用脚本：`utils/download_model_hf_home.sh`  
  - 在可联网环境中执行该脚本，将需要的模型下载到指定 `HF_HOME`；  
  - 将缓存目录同步到无法联网的 GPU 环境中；  
- 对于非 HuggingFace 模型：  
  - 请参考各模型官方提供的下载方式，利用可联网环境，手动下载到 `eval_models` 目录下或是 `$HOME/.cache` 的相应位置下，再同步到评测环境中；
- 在下载完整之后，请在 config.sh 中设置 `HOME`、`HF_HOME`、`TRANSFORMERS_CACHE` 以及 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，即可在离线模式下完成评测。

---

## 引用
如果您在研究中使用了 GenEditEvalKit，或希望参考已发布的开源评估结果，请使用以下 BibTeX 条目以及与您使用的特定模型 / 基准测试相对应的 BibTex 条目。
```bibtex
```

此外，我们也将持续支持更多模型和 benchmark ，并不断完善本工具包。同时也欢迎社区贡献 PR 🎉

> **我们欢迎以下类型的 PR：**
> - 支持 **新模型** 和 **新 benchmark** 的接入，
> - 修复 **bug** 和改进 **文档**，
> - 提升 **运行效率**、**稳定性** 和 **资源利用率**，
> - 以及增强工具包的 **模块化**、**可扩展性** 与整体 **易用性**。