_base_ = ['./segformer_mit-b5_512x512_160k_ade20k.py']

model = dict(decode_head=dict(type='SegformerHead_ssa'))