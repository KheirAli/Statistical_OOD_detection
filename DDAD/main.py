# import torch
# import numpy as np
# import os
# import argparse
# from unet import *
# from omegaconf import OmegaConf
# from train import trainer
# from feature_extractor import * 
# from ddad import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

# def build_model(config):
#     if config.model.DDADS:
#         unet = UNetModel(config.data.image_size, 32, dropout=0.3, n_heads=2 ,in_channels=config.data.input_channel)
#     else:
#         unet = UNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4 ,in_channels=config.data.input_channel)
#     return unet

# def train(config):
#     torch.manual_seed(42)
#     np.random.seed(42)
#     unet = build_model(config)
#     print(" Num params: ", sum(p.numel() for p in unet.parameters()))
#     unet = unet.to(config.model.device)
#     unet.train()
#     unet = torch.nn.DataParallel(unet)
#     # checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'1000'))
#     # unet.load_state_dict(checkpoint)  
#     trainer(unet, config.data.category, config)#config.data.category, 


# def detection(config):
#     unet = build_model(config)
#     checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
#     unet = torch.nn.DataParallel(unet)
#     unet.load_state_dict(checkpoint)    
#     unet.to(config.model.device)
#     checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
#     unet.eval()
#     ddad = DDAD(unet, config)
#     ddad()
    

# def finetuning(config):
#     unet = build_model(config)
#     checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)))
#     unet = torch.nn.DataParallel(unet)
#     unet.load_state_dict(checkpoint)    
#     unet.to(config.model.device)
#     unet.eval()
#     domain_adaptation(unet, config, fine_tune=True)
############## Before adding SEED setting and checkpoint_name override ##############
# import torch
# import numpy as np
# import os
# import argparse
# from omegaconf import OmegaConf
# import random
# from train import trainer
# from feature_extractor import *
# from ddad import *

# from unet import UNetModel as DDADUNetModel
# # from guided_diffusion.unet import UNetModel as GuidedUNetModel

# # os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# def _parse_channel_mult(v):
#     if isinstance(v, str):
#         return tuple(int(x) for x in v.split(","))
#     return tuple(v)


# def _parse_attention_ds(image_size, v):
#     # guided_diffusion.UNetModel expects downsample factors, not "16,8" strings
#     if isinstance(v, str):
#         return tuple(image_size // int(r) for r in v.split(","))
#     return tuple(v)


# def _unwrap_state_dict(ckpt):
#     # handle common checkpoint containers
#     if isinstance(ckpt, dict):
#         for k in ["state_dict", "model_state_dict", "model", "ema_state_dict", "ema"]:
#             if k in ckpt and isinstance(ckpt[k], dict):
#                 ckpt = ckpt[k]
#                 break

#     # strip DataParallel prefix if present
#     if isinstance(ckpt, dict) and len(ckpt) > 0:
#         first_key = next(iter(ckpt))
#         if first_key.startswith("module."):
#             ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

#     return ckpt


# def build_model(config):
#     if getattr(config.model, "use_external_unet", False):
#         m = config.model.external_unet

#         model = GuidedUNetModel(
#             image_size=config.data.image_size,
#             in_channels=config.data.input_channel,
#             model_channels=m.num_channels,
#             out_channels=config.data.input_channel * (2 if m.learn_sigma else 1),
#             num_res_blocks=m.num_res_blocks,
#             attention_resolutions=_parse_attention_ds(
#                 config.data.image_size, m.attention_resolutions
#             ),
#             dropout=m.dropout,
#             channel_mult=_parse_channel_mult(m.channel_mult),
#             num_classes=None,  # class_cond = False
#             use_checkpoint=m.use_checkpoint,
#             use_fp16=m.use_fp16,
#             num_heads=m.num_heads,
#             num_head_channels=m.num_head_channels,
#             num_heads_upsample=m.num_heads_upsample,
#             use_scale_shift_norm=m.use_scale_shift_norm,
#             resblock_updown=m.resblock_updown,
#             use_new_attention_order=m.use_new_attention_order,
#         )

#         state = torch.load(m.model_path, map_location="cpu")
#         state = _unwrap_state_dict(state)
#         model.load_state_dict(state, strict=True)
#         return model

#     # original DDAD path
#     if config.model.DDADS:
#         return DDADUNetModel(
#             config.data.image_size,
#             32,
#             dropout=0.3,
#             n_heads=2,
#             in_channels=config.data.input_channel,
#         )

#     return DDADUNetModel(
#         config.data.image_size,
#         64,
#         dropout=0.0,
#         n_heads=4,
#         in_channels=config.data.input_channel,
#     )


# def _maybe_parallel(model):
#     if torch.cuda.is_available() and torch.cuda.device_count() > 1:
#         return torch.nn.DataParallel(model)
#     return model


# def train(config):
#     torch.manual_seed(42)
#     np.random.seed(42)
#     unet = build_model(config)
#     print("Num params:", sum(p.numel() for p in unet.parameters()))
#     unet = unet.to(config.model.device)
#     unet.train()
#     unet = _maybe_parallel(unet)
#     trainer(unet, config.data.category, config)


# # def detection(config):
# #     unet = build_model(config)

# #     # only load DDAD-style checkpoint if NOT using external model
# #     if not getattr(config.model, "use_external_unet", False):
# #         checkpoint = torch.load(
# #             os.path.join(
# #                 os.getcwd(),
# #                 config.model.checkpoint_dir,
# #                 config.data.category,
# #                 str(config.model.load_chp),
# #             ),
# #             map_location="cpu",
# #         )
# #         checkpoint = _unwrap_state_dict(checkpoint)
# #         unet.load_state_dict(checkpoint, strict=True)

# #     unet = _maybe_parallel(unet).to(config.model.device)
# #     unet.eval()

# #     ddad = DDAD(unet, config)
# #     ddad()
# def detection(config):
#     unet = build_model(config)

#     if not getattr(config.model, "use_external_unet", False):
#         # Use checkpoint_name if set, otherwise build from parts
#         ckpt_path = getattr(config.model, "checkpoint_name", None) or os.path.join(
#             config.model.checkpoint_dir,
#             config.data.category,
#             str(config.model.load_chp),
#         )
#         print(f"Loading checkpoint: {ckpt_path}")
#         checkpoint = torch.load(ckpt_path, map_location="cpu")
#         checkpoint = _unwrap_state_dict(checkpoint)
#         unet.load_state_dict(checkpoint, strict=True)

#     unet = _maybe_parallel(unet).to(config.model.device)
#     unet.eval()
#     ddad = DDAD(unet, config)
#     ddad()


# def finetuning(config):
#     unet = build_model(config)

#     if not getattr(config.model, "use_external_unet", False):
#         checkpoint = torch.load(
#             os.path.join(
#                 os.getcwd(),
#                 config.model.checkpoint_dir,
#                 config.data.category,
#                 str(config.model.load_chp),
#             ),
#             map_location="cpu",
#         )
#         checkpoint = _unwrap_state_dict(checkpoint)
#         unet.load_state_dict(checkpoint, strict=True)

#     unet = _maybe_parallel(unet).to(config.model.device)
#     unet.eval()
#     domain_adaptation(unet, config, fine_tune=True)



# def parse_args():
#     cmdline_parser = argparse.ArgumentParser('DDAD')    
#     cmdline_parser.add_argument('-cfg', '--config', 
#                                 default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
#                                 help='config file')
#     cmdline_parser.add_argument('--train', 
#                                 default= False, 
#                                 help='Train the diffusion model')
#     cmdline_parser.add_argument('--detection', 
#                                 default= False, 
#                                 help='Detection anomalies')
#     cmdline_parser.add_argument('--domain_adaptation', 
#                                 default= False, 
#                                 help='Domain adaptation')
#     cmdline_parser.add_argument('--category',  default=None, help='Override config.data.category')
#     cmdline_parser.add_argument('--load_chp',  default=None, type=int, help='Override config.model.load_chp')
#     cmdline_parser.add_argument('--device',    default=None, help='Override config.model.device')
#     args, unknowns = cmdline_parser.parse_known_args()
#     return args


# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     args = parse_args()
#     config = OmegaConf.load(args.config)
#     print("Class: ",config.data.category, "   w:", config.model.w, "   v:", config.model.v, "   load_chp:", config.model.load_chp,   "   feature extractor:", config.model.feature_extractor,"         w_DA: ",config.model.w_DA,"         DLlambda: ",config.model.DLlambda)
#     print(f'{config.model.test_trajectoy_steps=} , {config.data.test_batch_size=}')
#     if args.category is not None:
#         config.data.category = args.category
#     if args.load_chp is not None:
#         config.model.load_chp = args.load_chp
#         # keep checkpoint_name in sync
#         config.model.checkpoint_name = os.path.join(
#             config.model.checkpoint_dir,
#             config.data.category,
#             str(args.load_chp)
#         )
#     if args.device is not None:
#         config.model.device = args.device

#     print("Class:", config.data.category, "  load_chp:", config.model.load_chp)
#     torch.manual_seed(42)
#     np.random.seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     if args.train:
#         print('Training...')
#         train(config)
#     if args.domain_adaptation:
#         print('Domain Adaptation...')
#         finetuning(config)
#     if args.detection:
#         print('Detecting Anomalies...')
#         detection(config)


import torch
import numpy as np
import os
import argparse
import random
from omegaconf import OmegaConf

from train import trainer
from feature_extractor import *
from ddad import *
from unet import UNetModel as DDADUNetModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_channel_mult(v):
    if isinstance(v, str):
        return tuple(int(x) for x in v.split(","))
    return tuple(v)


def _parse_attention_ds(image_size, v):
    if isinstance(v, str):
        return tuple(image_size // int(r) for r in v.split(","))
    return tuple(v)


def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "ema_state_dict", "ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if isinstance(ckpt, dict) and len(ckpt) > 0:
        first_key = next(iter(ckpt))
        if first_key.startswith("module."):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    return ckpt


def build_model(config):
    if getattr(config.model, "use_external_unet", False):
        m = config.model.external_unet
        model = GuidedUNetModel(
            image_size=config.data.image_size,
            in_channels=config.data.input_channel,
            model_channels=m.num_channels,
            out_channels=config.data.input_channel * (2 if m.learn_sigma else 1),
            num_res_blocks=m.num_res_blocks,
            attention_resolutions=_parse_attention_ds(config.data.image_size, m.attention_resolutions),
            dropout=m.dropout,
            channel_mult=_parse_channel_mult(m.channel_mult),
            num_classes=None,
            use_checkpoint=m.use_checkpoint,
            use_fp16=m.use_fp16,
            num_heads=m.num_heads,
            num_head_channels=m.num_head_channels,
            num_heads_upsample=m.num_heads_upsample,
            use_scale_shift_norm=m.use_scale_shift_norm,
            resblock_updown=m.resblock_updown,
            use_new_attention_order=m.use_new_attention_order,
        )
        state = torch.load(m.model_path, map_location="cpu")
        state = _unwrap_state_dict(state)
        model.load_state_dict(state, strict=True)
        return model

    if config.model.DDADS:
        return DDADUNetModel(config.data.image_size, 32, dropout=0.3, n_heads=2, in_channels=config.data.input_channel)

    return DDADUNetModel(config.data.image_size, 64, dropout=0.0, n_heads=4, in_channels=config.data.input_channel)


def _maybe_parallel(model):
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model)
    return model


def train(config):
    unet = build_model(config)
    print("Num params:", sum(p.numel() for p in unet.parameters()))
    unet = unet.to(config.model.device)
    unet.train()
    unet = _maybe_parallel(unet)
    trainer(unet, config.data.category, config)


def detection(config):
    unet = build_model(config)

    if not getattr(config.model, "use_external_unet", False):
        ckpt_path = getattr(config.model, "checkpoint_name", None) or os.path.join(
            config.model.checkpoint_dir,
            config.data.category,
            str(config.model.load_chp),
        )
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        checkpoint = _unwrap_state_dict(checkpoint)
        unet.load_state_dict(checkpoint, strict=True)

    unet = _maybe_parallel(unet).to(config.model.device)
    unet.eval()
    ddad = DDAD(unet, config)
    ddad()


def finetuning(config):
    unet = build_model(config)

    if not getattr(config.model, "use_external_unet", False):
        checkpoint = torch.load(
            os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category, str(config.model.load_chp)),
            map_location="cpu",
        )
        checkpoint = _unwrap_state_dict(checkpoint)
        unet.load_state_dict(checkpoint, strict=True)

    unet = _maybe_parallel(unet).to(config.model.device)
    unet.eval()
    domain_adaptation(unet, config, fine_tune=True)


# def parse_args():
#     cmdline_parser = argparse.ArgumentParser('DDAD')
#     cmdline_parser.add_argument('-cfg', '--config',
#                                 default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'),
#                                 help='config file')
#     cmdline_parser.add_argument('--train',            default=False,  help='Train the diffusion model')
#     cmdline_parser.add_argument('--detection',        default=False,  help='Detection anomalies')
#     cmdline_parser.add_argument('--domain_adaptation',default=False,  help='Domain adaptation')
#     cmdline_parser.add_argument('--category',         default=None,   help='Override config.data.category')
#     cmdline_parser.add_argument('--load_chp',         default=None,   type=int, help='Override config.model.load_chp')
#     cmdline_parser.add_argument('--device',           default=None,   help='Override config.model.device')
#     # ── NEW ──────────────────────────────────────────────────────────────────
#     cmdline_parser.add_argument('--seed',             default=42,     type=int, help='Global random seed')
#     # ─────────────────────────────────────────────────────────────────────────
#     args, _ = cmdline_parser.parse_known_args()
#     return args


# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     args = parse_args()

#     # ── Apply seed before everything else ────────────────────────────────────
#     set_seed(args.seed)
#     print(f"[Seed] {args.seed}")
#     # ─────────────────────────────────────────────────────────────────────────

#     config = OmegaConf.load(args.config)
#     print("Class:", config.data.category, "  w:", config.model.w, "  v:", config.model.v,
#           "  load_chp:", config.model.load_chp, "  feature_extractor:", config.model.feature_extractor,
#           "  w_DA:", config.model.w_DA, "  DLlambda:", config.model.DLlambda)
#     print(f'{config.model.test_trajectoy_steps=} , {config.data.test_batch_size=}')

#     if args.category is not None:
#         config.data.category = args.category
#     if args.load_chp is not None:
#         config.model.load_chp = args.load_chp
#         config.model.checkpoint_name = os.path.join(
#             config.model.checkpoint_dir, config.data.category, str(args.load_chp)
#         )
#     if args.device is not None:
#         config.model.device = args.device

#     print("Class:", config.data.category, "  load_chp:", config.model.load_chp)

#     if args.train:
#         print('Training...')
#         train(config)
#     if args.domain_adaptation:
#         print('Domain Adaptation...')
#         finetuning(config)
#     if args.detection:
#         print('Detecting Anomalies...')
#         detection(config)
def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD')
    cmdline_parser.add_argument('-cfg', '--config',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'),
                                help='config file')
    cmdline_parser.add_argument('--checkpoint_name', default=None,
                            help='Override config.model.checkpoint_name directly')
    cmdline_parser.add_argument('--train',             default=False, help='Train the diffusion model')
    cmdline_parser.add_argument('--detection',         default=False, help='Detection anomalies')
    cmdline_parser.add_argument('--domain_adaptation', default=False, help='Domain adaptation')
    cmdline_parser.add_argument('--category',          default=None,  help='Override config.data.category')
    cmdline_parser.add_argument('--load_chp',          default=None,  type=int, help='Override config.model.load_chp')
    cmdline_parser.add_argument('--device',            default=None,  help='Override config.model.device')
    cmdline_parser.add_argument('--seed',              default=42,    type=int, help='Global random seed')
    cmdline_parser.add_argument('--subcategory', default=None,
                            help='Restrict test set to one subfolder, e.g. scissors')
    # ── ADD THIS ─────────────────────────────────────────────────────────────
    cmdline_parser.add_argument('--checkpoint_dir',    default=None,  help='Override config.model.checkpoint_dir')
    # ─────────────────────────────────────────────────────────────────────────
    args, _ = cmdline_parser.parse_known_args()
    return args


if __name__ == "__main__":

    torch.cuda.empty_cache()
    args = parse_args()


    set_seed(args.seed)
    print(f"[Seed] {args.seed}")

    config = OmegaConf.load(args.config)
    print("Class:", config.data.category, "  w:", config.model.w, "  v:", config.model.v,
          "  load_chp:", config.model.load_chp, "  feature_extractor:", config.model.feature_extractor,
          "  w_DA:", config.model.w_DA, "  DLlambda:", config.model.DLlambda)
    print(f'{config.model.test_trajectoy_steps=} , {config.data.test_batch_size=}')
    if getattr(args, 'checkpoint_name', None) is not None:
        config.model.checkpoint_name = args.checkpoint_name
    else:
        # Always rebuild from the (possibly overridden) parts
        config.model.checkpoint_name = os.path.join(
            config.model.checkpoint_dir,
            config.data.category,
            str(config.model.load_chp)
        )
    if args.subcategory is not None:
        config.data.subcategory = args.subcategory
    # ── Apply ALL overrides first, then build checkpoint_name ONCE ───────────
    if args.category is not None:
        config.data.category = args.category

    if args.checkpoint_dir is not None:               # ← must be before load_chp
        config.model.checkpoint_dir = args.checkpoint_dir

    if args.load_chp is not None:
        config.model.load_chp = args.load_chp

    # Now checkpoint_dir is already the correct overridden value
    config.model.checkpoint_name = os.path.join(
        config.model.checkpoint_dir,
        config.data.category,
        str(config.model.load_chp)
    )
    # ─────────────────────────────────────────────────────────────────────────

    if args.device is not None:
        config.model.device = args.device

    print("Class:", config.data.category, "  load_chp:", config.model.load_chp)
    print("Checkpoint:", config.model.checkpoint_name)  # confirm the resolved path

    if args.train:
        print('Training...')
        train(config)
    if args.domain_adaptation:
        print('Domain Adaptation...')
        finetuning(config)
    if args.detection:
        print('Detecting Anomalies...')
        detection(config)