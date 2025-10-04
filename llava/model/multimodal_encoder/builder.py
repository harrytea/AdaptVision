from .clip_encoder import CLIPVisionTower
from .eva_encoder import EVACILPVisionTower
from .siglip_encoder import SigLipVisionTower
from .siglip2_encoder import Siglip2VisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if "openai" in vision_tower:
        vision_model = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return vision_model
    elif "eva_clip_g" in vision_tower:
        vision_model = EVACILPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return vision_model
    elif "siglip2" in vision_tower:
        vision_model = Siglip2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return vision_model
    elif "siglip" in vision_tower:
        vision_model = SigLipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return vision_model
    raise ValueError(f'Unknown vision tower: {vision_tower}')
