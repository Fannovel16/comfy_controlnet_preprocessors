import torch
import cv2
# https://github.com/advimman/lama

import yaml
from omegaconf import OmegaConf
import numpy as np

from einops import rearrange
from custom_nodes.comfy_controlnet_preprocessors.util import annotator_ckpts_path, HWC3, resize_image
from custom_nodes.comfy_controlnet_preprocessors.nodes.util import img_tensor_to_np, img_np_to_tensor
import comfy.model_management as model_management
import os

from custom_nodes.comfy_controlnet_preprocessors.v11.inpaint_lama.saicinpainting.training.trainers import load_checkpoint

class LamaInpainting:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetLama.pth"
        modelpath = os.path.join(annotator_ckpts_path, "ControlNetLama.pth")
        if not os.path.exists(modelpath):
            from custom_nodes.comfy_controlnet_preprocessors.util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        cfg = yaml.safe_load(open(config_path, 'rt'))
        cfg = OmegaConf.create(cfg)
        cfg.training_model.predict_only = True
        cfg.visualizer.kind = 'noop'
        self.model = load_checkpoint(cfg, os.path.abspath(modelpath), strict=False, map_location='cpu')
        self.model = self.model.to(model_management.get_torch_device())
        self.model.eval()

    def __call__(self, color, mask):
        color = np.ascontiguousarray(color).astype(np.float32) / 255.0
        mask = np.ascontiguousarray(mask).astype(np.float32) / 255.0
        with torch.no_grad():
            color = torch.from_numpy(color).float().to(model_management.get_torch_device())
            mask = torch.from_numpy(mask).float().to(model_management.get_torch_device())
            mask = (mask > 0.5).float()
            color = color * (1 - mask)
            image_feed = torch.cat([color, torch.mean(mask, dim=2, keepdim=True)], dim=2)
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')
            result = self.model(image_feed)[0]
            result = rearrange(result, 'c h w -> h w c')
            result = result * mask + color * (1 - mask)
            result *= 255.0
            return result.detach().cpu().numpy().clip(0, 255).astype(np.uint8)

def preprocess(img, mask, res=512):
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(img.shape[1], img.shape[2]), mode="bilinear")
    mask = mask.movedim(1,-1).expand((-1,-1,-1,3))
    img_list = img_tensor_to_np(img)
    mask_list = img_tensor_to_np(mask)

    out_img = []

    for i, img in enumerate(img_list):
        img = HWC3(img)
        mask = HWC3(mask_list[i])
        H, W = img.shape[:2]
        res = 256  # Always use 256 since lama is trained on 256

        img_res = resize_image(img, res)
        mask_res = resize_image(mask, res)
        model_lama = LamaInpainting()

        # applied auto inversion
        prd_color = model_lama(img_res, mask_res)
        prd_color = cv2.resize(prd_color, (W, H))

        alpha = mask.astype(np.float32) / 255.0
        fin_color = prd_color.astype(np.float32) * alpha + img.astype(np.float32) * (1 - alpha)
        fin_color = fin_color.clip(0, 255).astype(np.uint8)
        #np.concatenate([fin_color, mask], axis=2)
        out_img.append(cv2.resize(HWC3(fin_color), (W, H), interpolation=cv2.INTER_AREA))

    return (img_np_to_tensor(out_img),)