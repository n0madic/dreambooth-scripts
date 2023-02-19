#!/usr/bin/env python

import sys
import random
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
import gradio as gr


model_path = sys.argv[1]

pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
if is_xformers_available():
    pipe.enable_xformers_memory_efficient_attention()

def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, seed=0):
    seed = seed or random.SystemRandom().randint(0, 2**32 - 1)
    generator = torch.Generator(device='cuda').manual_seed(int(seed))
    with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=generator
            ).images

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Place your input prompt here")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            run = gr.Button(value="Generate")
            with gr.Row():
                num_samples = gr.Number(label="Number of Samples", value=4)
                guidance_scale = gr.Slider(0, 20, 8, step=0.5, label="Guidance Scale")
            with gr.Row():
                height = gr.Number(label="Height", value=512)
                width = gr.Number(label="Width", value=512)
            num_inference_steps = gr.Slider(label="Steps", value=24)
            seed = gr.Number(None, label="Seed", precision=0)
        with gr.Column():
            gallery = gr.Gallery()

    run.click(inference, inputs=[prompt, negative_prompt, num_samples, height, width, num_inference_steps, guidance_scale, seed], outputs=gallery)

demo.launch(debug=True, share=True)
