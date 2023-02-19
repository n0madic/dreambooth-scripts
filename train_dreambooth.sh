#!/bin/bash

# Check $INSTANCE_NAME or $CLASS_PROMPT is set
if [ -z "$INSTANCE_NAME" ] || [ -z "$CLASS_PROMPT" ]; then
  echo "Please set INSTANCE_NAME and CLASS_PROMPT environment variables."
  exit 1
fi

# Set variables

PRETRAINED_MODEL_NAME="runwayml/stable-diffusion-v1-5"
INSTANCE_DIR="training/${INSTANCE_NAME}"
INSTANCE_DATA_DIR="${INSTANCE_DIR}/data/instance"
OUTPUT_DIR="${INSTANCE_DIR}/weights"
INSTANCE_PROMPT="${INSTANCE_NAME//-/ }"
CLASS_DATA_DIR="${INSTANCE_DIR}/data/class"
SAVE_SAMPLE_PROMPT="photo of ${INSTANCE_PROMPT}"
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-3000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}

# Activate virtual environment if exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Create directories
mkdir -p "${INSTANCE_DIR}" "${INSTANCE_DATA_DIR}" "${CLASS_DATA_DIR}" "${OUTPUT_DIR}"

# Check if Instance data dir empty and fail with error
if [ -z "$(ls -A ${INSTANCE_DATA_DIR})" ]; then
  echo "Instance data dir is empty. Please add images to ${INSTANCE_DATA_DIR}"
  exit 1
fi

# Launch training
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="${PRETRAINED_MODEL_NAME}" \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir="${OUTPUT_DIR}" \
  --revision="fp16" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --sample_batch_size=4 \
  --save_interval=${SAVE_INTERVAL} \
  --max_train_steps=${MAX_TRAIN_STEPS} \
  --instance_data_dir="${INSTANCE_DATA_DIR}" \
  --class_data_dir="${CLASS_DATA_DIR}" \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --class_prompt="${CLASS_PROMPT}" \
  --save_sample_prompt="${SAVE_SAMPLE_PROMPT}"

# Generate grid of sample images
python grid_generate.py "${OUTPUT_DIR}"

# Convert diffusers to checkpoint format
python convert_diffusers_to_original_stable_diffusion.py --half \
  --model_path="${OUTPUT_DIR}/${MAX_TRAIN_STEPS}" \
  --checkpoint_path="${OUTPUT_DIR}/${MAX_TRAIN_STEPS}/${INSTANCE_NAME}${MAX_TRAIN_STEPS}.ckpt"
