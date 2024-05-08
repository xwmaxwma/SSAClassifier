_base_ = ['./segformer_mit-b5_480x480_80k_pc.py']

model = dict(decode_head=dict(type='SegformerHead_ssa'))