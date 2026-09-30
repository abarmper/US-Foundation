"""Throughput micro-benchmark for the GRNET/AWS proposal (synthetic inputs, 1x A100).

Replicates the exact Phase-1 DINOv2 SSL step of gubiometry/engine/phase1_dinov2.py
(teacher fwd on 2 globals + heads, student fwd on masked globals + 6 locals, DINO + iBOT
+ KoLeo losses, grad-accum, AdamW step, EMA update) and the Phase-2 step (multilevel
HRNet neck, last-4 blocks unfrozen, soft-argmax L1) for ViT-B/L/g-14-reg, so the
model-size scaling can be read against the measured production throughput of ViT-L.
Weights are random (pretrained=False): throughput does not depend on weight values.
"""
import json, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/abar/dino2/dinov2-hrnet-gu-biometry-main")
import gubiometry.engine.common  # noqa: F401  (sets TORCH_HOME)
import gubiometry.engine.phase1_dinov2 as p1mod
import gubiometry.models.model as mmod
from gubiometry.config import RunConfig
from gubiometry.models.dino_ssl import DINOLossV2, iBOTPatchLossV2, KoLeoLoss
from gubiometry.geometry import soft_argmax_coords

HUB = "/data/abar/.cache/torch_gu_biometry/hub/facebookresearch_dinov2_main"


def load_backbone_random(name):
    enc = torch.hub.load(HUB, name, source="local", pretrained=False)
    return enc, enc.embed_dim


p1mod.load_backbone = load_backbone_random
mmod.load_backbone = load_backbone_random
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True


def ssl_bench(backbone, gsize, bs, accum, n_eff=2, warm_eff=1, lsize=98, n_local=6):
    cfg = RunConfig()
    p1 = cfg.phase1
    student = p1mod._DINOv2Wrapper(backbone, p1).to(dev)
    teacher = p1mod._DINOv2Wrapper(backbone, p1).to(dev)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval(); student.train()
    dino_loss = DINOLossV2(p1.dino_out_dim, student_temp=p1.student_temp, center_momentum=p1.center_momentum).to(dev)
    ibot_loss = iBOTPatchLossV2(p1.dino_out_dim, student_temp=p1.student_temp, center_momentum=p1.center_momentum).to(dev)
    koleo_loss = KoLeoLoss().to(dev)
    opt = torch.optim.AdamW(p1mod._build_param_groups(student, 0.04), lr=1e-4)
    n_tok = (gsize // 14) ** 2
    g = torch.randn(2 * bs, 3, gsize, gsize, device=dev)
    l = torch.randn(n_local * bs, 3, lsize, lsize, device=dev)
    rng = np.random.default_rng(0)
    masks = torch.zeros(2 * bs, n_tok, dtype=torch.bool)
    for j in range(2 * bs):
        if rng.random() < 0.5:
            k = int(n_tok * rng.uniform(0.1, 0.5))
            masks[j, torch.randperm(n_tok)[:k]] = True
    flat = masks.reshape(-1)
    mask_idx = torch.nonzero(flat).squeeze(1).to(dev)
    counts = masks.sum(1).clamp(min=1).float()
    mw = (1.0 / counts).unsqueeze(1).expand_as(masks)[masks].to(dev)
    masks = masks.to(dev)
    torch.cuda.reset_peak_memory_stats()

    def micro():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                t_out = teacher.encoder.forward_features(g)
                t_cls = teacher.dino_head(t_out["x_norm_clstoken"])
                t_soft = [dino_loss.softmax_center_teacher(c, 0.07) for c in t_cls.chunk(2)]
                tp = t_out["x_norm_patchtokens"].reshape(-1, t_out["x_norm_patchtokens"].shape[-1])
                t_mh = teacher.ibot_head(tp[mask_idx])
                t_isoft = ibot_loss.softmax_center_teacher(t_mh, 0.07)
            s_g = student.encoder.forward_features(g, masks=masks)
            s_cls = s_g["x_norm_clstoken"]
            s_dino = list(student.dino_head(s_cls).chunk(2))
            s_l = student.encoder.forward_features(l)
            s_dino += list(student.dino_head(s_l["x_norm_clstoken"]).chunk(n_local))
            dl = dino_loss.forward(s_dino, t_soft, skip_diagonal=True)
            sp = s_g["x_norm_patchtokens"].reshape(-1, s_g["x_norm_patchtokens"].shape[-1])
            il = ibot_loss.forward_masked(student.ibot_head(sp[mask_idx]), t_isoft, mw, 2 * bs)
            kl = sum(koleo_loss(c) for c in s_cls.chunk(2)) / 2.0
            tot = dl + il + 0.1 * kl
        (tot / accum).backward()
        return t_cls

    def eff_step():
        for _ in range(accum):
            t_cls = micro()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        dino_loss.update_center_ema(t_cls.detach().float().mean(0, keepdim=True))
        with torch.no_grad():
            for ps, pt in zip(student.parameters(), teacher.parameters()):
                pt.mul_(0.999).add_(ps.detach(), alpha=0.001)

    for _ in range(warm_eff):
        eff_step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n_eff):
        eff_step()
    torch.cuda.synchronize(); dt = time.time() - t0
    imgs = n_eff * accum * bs
    out = dict(kind="ssl", backbone=backbone, res=gsize, bs=bs, accum=accum,
               img_s=imgs / dt, peak_gb=torch.cuda.max_memory_allocated() / 2**30,
               params_M=sum(p.numel() for p in student.encoder.parameters()) / 1e6)
    del student, teacher, opt
    torch.cuda.empty_cache()
    return out


def p2_bench(backbone, layers, bs=64, n_steps=8, warm=3, unfreeze=4):
    m = mmod.UnifiedBiometryModel(backbone_name=backbone, freeze_encoder=True, heatmap_size=148,
                                  unfreeze_last_n_blocks=unfreeze, input_mode="multilevel",
                                  feature_layers=layers, neck_decoder="hrnet").to(dev)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4)
    x = torch.randn(bs, 3, 518, 518, device=dev)
    gt = torch.rand(bs, 4, 2, device=dev) * 518
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m.forward_phase2(x, "HC").float()
            coords = soft_argmax_coords(logits, 10.0, 518)
            loss = F.l1_loss(coords, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

    for _ in range(warm):
        step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize(); dt = time.time() - t0
    out = dict(kind="phase2", backbone=backbone, res=518, bs=bs, unfreeze=unfreeze,
               img_s=n_steps * bs / dt, peak_gb=torch.cuda.max_memory_allocated() / 2**30,
               params_M=sum(p.numel() for p in m.parameters()) / 1e6,
               trainable_M=sum(p.numel() for p in params) / 1e6)
    del m, opt
    torch.cuda.empty_cache()
    return out


def safe(fn, *a, **k):
    try:
        r = fn(*a, **k)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        r = dict(error="OOM", args=[str(x) for x in a], kw=k)
    print(json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "ssl":
        safe(ssl_bench, "dinov2_vitl14_reg", 224, 32, 8)      # production config (measured 64.4 img/s)
        safe(ssl_bench, "dinov2_vitl14_reg", 518, 16, 16)     # production tail (measured 15.9 img/s)
        safe(ssl_bench, "dinov2_vitb14_reg", 224, 32, 8)
        safe(ssl_bench, "dinov2_vitb14_reg", 518, 16, 16)
        r = safe(ssl_bench, "dinov2_vitg14_reg", 224, 32, 8)
        if "error" in r:
            safe(ssl_bench, "dinov2_vitg14_reg", 224, 16, 16)
        r = safe(ssl_bench, "dinov2_vitg14_reg", 518, 8, 32, n_eff=1)
        if "error" in r:
            safe(ssl_bench, "dinov2_vitg14_reg", 518, 4, 64, n_eff=1)
    else:
        safe(p2_bench, "dinov2_vitl14_reg", (5, 11, 17, 23))  # production (measured 84.4 img/s)
        safe(p2_bench, "dinov2_vitb14_reg", (2, 5, 8, 11))
        r = safe(p2_bench, "dinov2_vitg14_reg", (9, 19, 29, 39))
        if "error" in r:
            safe(p2_bench, "dinov2_vitg14_reg", (9, 19, 29, 39), bs=32)
