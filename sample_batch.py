from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--box_coords', type=int, nargs=4, default=None,
                    help='Box coordinates for refined_box mask: top bottom left right')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default=None,
                help='Override data root directory path')
    parser.add_argument('--mask_prob', type=float, default=1.0,
                    help='Mask probability range (0.0-1.0)')
    parser.add_argument('--mask_path', type=str, default="/home/akheirandish3/diffusion-posterior-sampling/data/mask.png",
                    help='Path to custom mask image')
    parser.add_argument('--num_measurements', type=int, default=1,
                    help='Number of different measurements (masks) per image')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    seed = args.seed
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    if args.data_root:
        task_config['data']['root'] = args.data_root

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    # if measure_config['operator']['name'] == 'inpainting':
    #     mask_gen = mask_generator(
    #        **measure_config['mask_opt']
    #     )

    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask_opt = measure_config['mask_opt']
            if mask_opt.get('mask_type') == 'refined_box':
                assert args.box_coords is not None, \
                    "box_coords required for refined_box mask type. Use --box_coords top bottom left right"
                box_coords = tuple(args.box_coords)
                mask_gen = mask_generator(**mask_opt)

                masks = []
                for _ in range(args.num_measurements):
                    m = mask_gen(ref_img, box_coords=box_coords,
                                 box_prob_multiplier=args.mask_prob,
                                 mask_path=args.mask_path)
                    m = m[:, 0, :, :].unsqueeze(1)  # (1,1,H,W)
                    masks.append(m)
                mask = torch.cat(masks, dim=0)  # (M,1,H,W)
            else:
                mask_gen = mask_generator(**mask_opt)

                masks = []
                for _ in range(args.num_measurements):
                    m = mask_gen(ref_img)
                    m = m[:, 0, :, :].unsqueeze(1)  # (1,1,H,W)
                    masks.append(m)
                mask = torch.cat(masks, dim=0)  # (M,1,H,W)
            if args.num_measurements > 1:
                ref_img = ref_img.repeat(args.num_measurements, 1, 1, 1)

            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)
        batch = ref_img.shape[0]
        for b in range(batch):
            suffix = f"{b}_" if batch > 1 else ""
            plt.imsave(os.path.join(out_path, 'input', f"{suffix}{fname}"), clear_color(y_n[b]))
            plt.imsave(os.path.join(out_path, 'label', f"{suffix}{fname}"), clear_color(ref_img[b]))
            plt.imsave(os.path.join(out_path, 'recon', f"{suffix}{seed}_{fname}"), clear_color(sample[b]))
        break  # --- REMOVE THIS LINE TO PROCESS THE WHOLE DATASET ---

if __name__ == '__main__':
    main()
