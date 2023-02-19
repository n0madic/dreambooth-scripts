#!/usr/bin/env python

import argparse
import random
import os


def compare_models(seeds, model_list, prompt, height, width, steps, scale):
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from diffusers.utils.import_utils import is_xformers_available

    print(f"Prompt: {prompt}")
    results = {}
    for model in model_list:
        print(f"Model: {model}")
        pipe = StableDiffusionPipeline.from_pretrained(model, requires_safety_checker=False, torch_dtype=torch.float16).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()
        for seed in seeds:
            print(f"Seed: {seed}")
            if seed not in results:
                results[seed] = {}
            generator = torch.Generator(device='cuda').manual_seed(int(seed))
            with torch.autocast("cuda"), torch.inference_mode():
                results[seed][model] = pipe(
                        prompt,
                        height=height, width=width,
                        num_images_per_prompt=1,
                        num_inference_steps=steps,
                        guidance_scale=scale,
                        generator=generator
                    ).images[0]
    return results


def save_grid(results, seeds, models, output_file):
    import matplotlib.pyplot as plt

    row = len(seeds)
    col = len(models)
    scale = 4
    fig, axes = plt.subplots(row, col, figsize=(col*scale, row*scale), gridspec_kw={'hspace': 0, 'wspace': 0})
    for i, seed in enumerate(seeds):
        for j, model in enumerate(models):
            if row == 1:
                currAxes = axes[j]
            else:
                currAxes = axes[i, j]
            if i == 0:
                basename = os.path.basename(model)
                currAxes.set_title(basename)
            if j == 0:
                currAxes.text(-0.2, 0.5, seed, rotation=0, va='center', ha='center', transform=currAxes.transAxes)
            currAxes.imshow(results[seed][model], cmap='gray')
            currAxes.axis('off')
    plt.tight_layout()
    plt.savefig(output_file)
    print("Saved {}".format(output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model compare utility')
    parser.add_argument('--models', metavar='PATH', type=str, nargs='+',
                        help='list of model paths')
    parser.add_argument('--prompt', type=str, required=True,
                        help='prompt for comparing')
    parser.add_argument('--height', type=int, default=512,
                        help='image height (default: 512)')
    parser.add_argument('--width', type=int, default=512,
                        help='image width (default: 512)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='list of seeds to use')
    parser.add_argument('--count', type=int, default=3,
                        help='number of seeds to generate (default: 3)')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of inference steps (default: 50)')
    parser.add_argument('--scale', type=float, default=7.5,
                        help='guidance scale (default: 7.5)')
    parser.add_argument('--output', type=str, default="model_compare.png",
                        help='output file name')

    args = parser.parse_args()

    if not args.seeds:
        seeds = [random.SystemRandom().randint(0, 2**32 - 1) for _ in range(args.count)]

    results = compare_models(seeds, args.models, args.prompt,
                             args.height, args.width,
                             args.steps, args.scale)
    save_grid(results, seeds, args.models, args.output)
