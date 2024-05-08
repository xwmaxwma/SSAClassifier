from .ocr_head_ssa import OCRHead_ssa
from .uper_head_ssa import UPerHead_ssa
from .segformer_head_ssa import SegformerHead_ssa
from .light_head import LightHead
from .light_head_ssa import LightHead_ssa
from .ham_head import LightHamHead
from .ham_head_ssa import LightHamHead_ssa
from .aff_head import CLS
from .aff_head_ssa import CLS_ssa


__all__ = [
    'OCRHead_ssa', 'UPerHead_ssa', 'SegformerHead_ssa', 'LightHead', 'LightHead_ssa', 'LightHamHead', 'LightHamHead_ssa','CLS', 'CLS_ssa'
]
