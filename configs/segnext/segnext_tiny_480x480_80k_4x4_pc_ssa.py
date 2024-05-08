_base_ = ['segnext_tiny_480x480_80k_4x4_pc.py']

model = dict(
    decode_head=dict(type='LightHamHead_ssa')
)