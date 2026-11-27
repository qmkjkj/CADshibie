"""Optional learned residual on RGB and instance coarse ink at native pixels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .detail_unet import NativeDetailUNet
from .instance_prompt import coordinate_prompt


def tile_positions(length: int, size: int, overlap: int = 64):
    if length <= 0 or size <= 0:
        raise ValueError("Tile and source dimensions must be positive")
    overlap = min(max(overlap, 0), size//2)
    if length <= size:
        return [0]
    return sorted(set(list(range(0, length-size+1, size-overlap)) + [length-size]))


class NativeDetailRefiner:
    def __init__(self, model, device, tile=320, classes=None):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.tile = int(tile)
        self.classes = None if classes is None else set(map(int, classes))
        self.uses_instance_prompt = getattr(model, "conditioning", "none") == "instance"

    @classmethod
    def load(cls, path, device="cuda"):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        architecture = checkpoint.get("architecture")
        if architecture not in ("native-residual-v1", "native-residual-v2", "native-residual-v3"):
            raise ValueError("Expected a native residual detail checkpoint")
        if (checkpoint.get("architecture") == "native-residual-v1"
                and checkpoint["model_kwargs"].get("prior_mode", "fixed") != "fixed"):
            raise ValueError("A boundary prior requires a version 2 checkpoint")
        conditional = checkpoint["model_kwargs"].get("conditioning", "none") == "instance"
        if conditional != (architecture == "native-residual-v3"):
            raise ValueError("Instance conditioning requires a version 3 checkpoint")
        model = NativeDetailUNet(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, device, checkpoint["tile"], checkpoint.get("classes"))

    @torch.no_grad()
    def correct(self, image, coarse, class_id, *, det_box=None):
        if self.classes is not None and class_id not in self.classes:
            return coarse
        height, width = coarse.shape
        if image.shape != (height, width, 3):
            raise ValueError("Native RGB and coarse probability have different frames")
        if not np.isfinite(coarse).all():
            raise ValueError("Nonfinite coarse probabilities")
        if self.uses_instance_prompt and det_box is None:
            raise ValueError("Instance detail requires its detector box in the crop frame")
        result = np.zeros(coarse.shape, dtype=np.float32)
        counts = np.zeros(coarse.shape, dtype=np.float32)
        # Keep the coarse channel in float32 at inference; stored training
        # probabilities are quantized only as an explicit dataset compression.
        positions = [(x,y) for y in tile_positions(height, self.tile)
                     for x in tile_positions(width, self.tile)]
        for start in range(0, len(positions), 8):
            batch = []; metadata = []; prompts = []
            for x,y in positions[start:start+8]:
                valid_w, valid_h = min(width-x, self.tile), min(height-y, self.tile)
                sample = np.ones((self.tile, self.tile, 4), dtype=np.float32)
                sample[..., 3] = 0
                sample[:valid_h, :valid_w, :3] = image[y:y+valid_h, x:x+valid_w] / 255.
                sample[:valid_h, :valid_w, 3] = coarse[y:y+valid_h, x:x+valid_w]
                batch.append(sample); metadata.append((x,y,valid_w,valid_h))
                if self.uses_instance_prompt:
                    prompts.append(coordinate_prompt((self.tile,self.tile),det_box,(x,y),
                                                     (valid_h,valid_w)))
            tensor = torch.from_numpy(np.stack(batch)).permute(0,3,1,2).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                if self.uses_instance_prompt:
                    class_ids = torch.full((len(batch),),int(class_id),dtype=torch.long,device=self.device)
                    prompt = torch.from_numpy(np.stack(prompts)).to(self.device)
                    logits = self.model(tensor,class_ids,prompt)
                else:
                    logits = self.model(tensor)
                correction = torch.sigmoid(logits.float()).cpu().numpy()[:,0]
            for probability, (x,y,valid_w,valid_h) in zip(correction, metadata):
                result[y:y+valid_h, x:x+valid_w] += probability[:valid_h, :valid_w]
                counts[y:y+valid_h, x:x+valid_w] += 1
        corrected = result / np.maximum(counts, 1)
        if not np.isfinite(corrected).all():
            return coarse
        return corrected
