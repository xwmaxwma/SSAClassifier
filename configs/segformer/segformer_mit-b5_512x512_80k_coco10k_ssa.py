_base_ = ['./segformer_mit-b5_512x512_80k_coco10k.py']

model = dict(decode_head=dict(type='SegformerHead_ssa'))