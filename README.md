# DreamBooth scripts

This repository contains the scripts used to DreamBooth training.

## Requirements

- Python 3
- [diffusers](https://github.com/huggingface/diffusers)
- [gradio](https://gradio.app/) for Web UI

## Installation

```bash
git clone https://github.com/n0madic/dreambooth-scripts.git
cd dreambooth-scripts
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download additional scripts:

```bash
wget -q https://github.com/huggingface/diffusers/raw/main/examples/dreambooth/train_dreambooth.py
wget -q https://github.com/huggingface/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py
```

## Usage

### Launch training

```bash
INSTANCE_NAME="<token>" CLASS_PROMPT="photo of woman" ./train_dreambooth.sh
```

### Generate grid

```bash
./grid_generate.py <PATH_TO_WEIGHTS>
```

### Compare models

```bash
./model_compare.py --prompt "photo of <token>" --model <PATH1> <PATH2> <PATH3> ...
```

### Run Web UI for model testing

```bash
./webui.py <PATH_TO_MODEL>
```
