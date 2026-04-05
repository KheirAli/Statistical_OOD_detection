"""DPS sampling orchestration — wraps sample_batch.py as subprocess.

Generates temporary config files from the unified experiment.yaml,
then calls sample_batch.py with the right CLI args for each patch.

IMPORTANT: sample_batch.py's --box_coords are NOT pixel coordinates.
They are (superpixel_index, mode_flag, 0, 0) where mode_flag=4 means
"random probability masking based on superpixel label map". This matches
how run_with_mask.sh invokes it.
"""

import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import yaml

# Path to the DPS repo (provides guided_diffusion and data.dataloader)
DPS_REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "dps")
DPS_REPO_DIR = os.path.normpath(DPS_REPO_DIR)

# Mode flag for refined_box mask: 4 = random probability mask based on superpixel map
MASK_MODE = 4


def write_temp_configs(
    sampling_cfg: Dict, data_cfg: Dict
) -> Dict[str, str]:
    """Generate temporary diffusion and task config files.

    Returns dict with keys: model_config, diffusion_config, task_config.
    """
    model_config_path = sampling_cfg["model"]["config"]

    # If a checkpoint override is specified, create a temp model config
    if sampling_cfg["model"].get("checkpoint"):
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f)
        model_cfg["model_path"] = sampling_cfg["model"]["checkpoint"]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(model_cfg, tmp)
        tmp.close()
        model_config_path = tmp.name

    # Diffusion config
    diff = sampling_cfg["diffusion"]
    diffusion_cfg = {
        "sampler": diff["sampler"],
        "steps": diff["steps"],
        "noise_schedule": diff["noise_schedule"],
        "model_mean_type": diff["model_mean_type"],
        "model_var_type": diff["model_var_type"],
        "dynamic_threshold": False,
        "clip_denoised": True,
        "rescale_timesteps": False,
        "timestep_respacing": diff.get("timestep_respacing", str(diff["steps"])),
    }
    diff_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(diffusion_cfg, diff_tmp)
    diff_tmp.close()

    # Task config (inpainting)
    mask_cfg = sampling_cfg["measurement"]["mask"]
    task_cfg = {
        "conditioning": {
            "method": sampling_cfg["conditioning"]["method"],
            "params": {"scale": sampling_cfg["conditioning"]["scale"]},
        },
        "data": {
            "name": "ffhq",
            "root": data_cfg["image_dir"],
        },
        "measurement": {
            "operator": {"name": sampling_cfg["measurement"]["operator"]},
            "mask_opt": {
                "mask_type": mask_cfg["type"],
                "mask_len_range": tuple(mask_cfg["mask_len_range"]),
                "image_size": mask_cfg["image_size"],
            },
            "noise": {
                "name": sampling_cfg["measurement"]["noise"]["type"],
                "sigma": sampling_cfg["measurement"]["noise"]["sigma"],
            },
        },
    }
    task_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(task_cfg, task_tmp)
    task_tmp.close()

    return {
        "model_config": model_config_path,
        "diffusion_config": diff_tmp.name,
        "task_config": task_tmp.name,
    }


def build_sample_command(
    config_paths: Dict[str, str],
    save_dir: str,
    data_root: str,
    superpixel_index: int,
    gpu_id: int,
    seed: int,
    mask_prob: float,
    mask_path: str,
    num_measurements: int,
) -> List[str]:
    """Build the sample_batch.py CLI command.

    box_coords = (superpixel_index, MASK_MODE, 0, 0):
      - superpixel_index: which superpixel label to focus masking on
      - MASK_MODE=4: use random probability mask (the active code path)
      - last two args: unused in mode 4
    """
    return [
        sys.executable, "sample_batch.py",
        f"--model_config={config_paths['model_config']}",
        f"--diffusion_config={config_paths['diffusion_config']}",
        f"--task_config={config_paths['task_config']}",
        f"--save_dir={save_dir}",
        f"--data_root={data_root}",
        f"--seed={seed}",
        "--box_coords", str(superpixel_index), str(MASK_MODE), "0", "0",
        f"--mask_prob={mask_prob}",
        f"--mask_path={mask_path}",
        f"--num_measurements={num_measurements}",
        f"--gpu={gpu_id}",
    ]


def run_sampling(sampling_cfg: Dict, data_cfg: Dict) -> str:
    """Orchestrate DPS sampling across all patches.

    For each superpixel index 0..num_patches-1, runs sample_batch.py
    with a probability mask focused on that superpixel region.

    Returns:
        Path to results directory.
    """
    if not sampling_cfg.get("enabled", False):
        print("Sampling disabled, using existing results.")
        return data_cfg["results_dir"]

    sample_name = data_cfg["sample_name"]
    results_dir = data_cfg["results_dir"]
    figures_dir = data_cfg["figures_dir"]
    image_dir = data_cfg["image_dir"]
    test_origin = data_cfg["test_origin"]
    bottom_suffix = data_cfg["bottom_suffix"]

    config_paths = write_temp_configs(sampling_cfg, data_cfg)

    # Build env with DPS on PYTHONPATH so sample_batch.py can import guided_diffusion
    dps_dir = sampling_cfg.get("dps_repo_dir", DPS_REPO_DIR)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{dps_dir}:{existing}" if existing else dps_dir

    # Generate superpixels for this sample's image
    sample_id = sample_name.split("_", 1)[1] if "_" in sample_name else sample_name
    input_image = os.path.join(image_dir, f"{sample_id}.png")
    os.makedirs(figures_dir, exist_ok=True)
    if os.path.exists(input_image):
        print(f"Generating superpixels for {input_image}...")
        subprocess.run(
            [sys.executable, "super_pixel_generation.py",
             f"--input_image={input_image}",
             f"--output_dir={figures_dir}"],
            check=True, env=env,
        )

    # Run sampling per superpixel patch
    num_patches = sampling_cfg.get("num_patches", 24)
    gpu_ids = sampling_cfg.get("gpu_ids", [0])
    seed = sampling_cfg.get("seed", 0)
    mask_prob = sampling_cfg["measurement"]["mask"]["mask_prob"]
    mask_path = os.path.join(os.path.abspath(figures_dir), "mask.png")
    num_measurements = sampling_cfg.get("num_measurements", 4)

    processes: List[subprocess.Popen] = []

    for patch_idx in range(num_patches):
        gpu_id = gpu_ids[patch_idx % len(gpu_ids)]
        save_dir = os.path.join(
            results_dir, sample_name,
            f"{test_origin}_{patch_idx}_{bottom_suffix}",
        )
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = build_sample_command(
            config_paths=config_paths,
            save_dir=save_dir,
            data_root=image_dir,
            superpixel_index=patch_idx,
            gpu_id=0,  # always 0 since CUDA_VISIBLE_DEVICES handles mapping
            seed=seed,
            mask_prob=mask_prob,
            mask_path=mask_path,
            num_measurements=num_measurements,
        )

        print(f"  Patch {patch_idx} on GPU {gpu_id}...")
        proc = subprocess.Popen(cmd, env=env)
        processes.append(proc)

        # Wait for batch if we've filled all GPUs
        if len(processes) >= len(gpu_ids):
            for p in processes:
                p.wait()
            processes = []

    # Wait for remaining
    for p in processes:
        p.wait()

    # Cleanup temp configs
    for path in config_paths.values():
        if path.startswith(tempfile.gettempdir()):
            os.unlink(path)

    print(f"Sampling complete. Results in {results_dir}/{sample_name}/")
    return results_dir
