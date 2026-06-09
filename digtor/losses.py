import torch
import torch.nn.functional as F

from . import NUM_CLASSES, IGNORE_INDEX


def ce_loss(logits, target, weight=None, ignore_index=IGNORE_INDEX):
    return F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index)


def dice_loss(logits, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, eps=1e-6):
    """Soft multi-class Dice over valid pixels."""
    p = F.softmax(logits, dim=1)
    valid = (target != ignore_index)
    t = target.clamp(0, num_classes - 1)
    onehot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
    vmask = valid.unsqueeze(1).float()
    p = p * vmask
    onehot = onehot * vmask
    inter = (p * onehot).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def seg_loss(logits, target, dice_weight=1.0, ce_weight=1.0,
             class_weight=None, ignore_index=IGNORE_INDEX):
    """CE + Dice. Dice stabilises the heavily imbalanced FMB class frequencies
    (road/sky dominate; person/pole are tiny)."""
    ce = ce_loss(logits, target, weight=class_weight, ignore_index=ignore_index)
    dc = dice_loss(logits, target, num_classes=logits.shape[1], ignore_index=ignore_index)
    return ce_weight * ce + dice_weight * dc


def reliability_loss(c_v, c_t, logit_v, logit_t, target, ignore_index=IGNORE_INDEX):
    """BCE between per-modality reliability and the per-pixel class-correctness
    oracle, downsampled to the reliability-map resolution."""
    with torch.no_grad():
        valid = (target != ignore_index).float()
        av = (logit_v.argmax(1) == target).float() * valid
        at = (logit_t.argmax(1) == target).float() * valid
        av_lr = F.adaptive_avg_pool2d(av.unsqueeze(1), c_v.shape[-2:])
        at_lr = F.adaptive_avg_pool2d(at.unsqueeze(1), c_t.shape[-2:])
    # F.binary_cross_entropy is unsafe under autocast (AMP); run it in fp32 with
    # autocast disabled. clamp keeps log() in BCE away from 0/1 (no NaN).
    with torch.autocast(device_type=c_v.device.type, enabled=False):
        cv = c_v.float().clamp(1e-6, 1.0 - 1e-6)
        ct = c_t.float().clamp(1e-6, 1.0 - 1e-6)
        return (F.binary_cross_entropy(cv, av_lr.float())
                + F.binary_cross_entropy(ct, at_lr.float()))


def ms_reliability_loss(cv_s, ct_s, logit_v, logit_t, target, ignore_index=IGNORE_INDEX):
    """Multi-scale version of reliability_loss for the hierarchical router.

    Each scale's V/T reliability head is supervised against the correctness
    oracle downsampled to that scale. Without it the heads output a near-constant
    map and the refiner cannot localise the (sparse) thermal-rescue region."""
    loss = 0.0
    with torch.no_grad():
        valid = (target != ignore_index).float()
        av = (logit_v.argmax(1) == target).float() * valid
        at = (logit_t.argmax(1) == target).float() * valid
    for cv, ct in zip(cv_s, ct_s):
        if cv is None:
            continue
        with torch.no_grad():
            av_lr = F.adaptive_avg_pool2d(av.unsqueeze(1), cv.shape[-2:])
            at_lr = F.adaptive_avg_pool2d(at.unsqueeze(1), ct.shape[-2:])
        with torch.autocast(device_type=cv.device.type, enabled=False):
            cvf = cv.float().clamp(1e-6, 1.0 - 1e-6)
            ctf = ct.float().clamp(1e-6, 1.0 - 1e-6)
            loss = loss + (F.binary_cross_entropy(cvf, av_lr.float())
                           + F.binary_cross_entropy(ctf, at_lr.float()))
    n = sum(1 for c in cv_s if c is not None)
    return loss / max(n, 1)


def task_align_loss(pv, pt, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """Cross-modal class-prototype alignment.

    Pulls same-class V/T prototypes together and pushes different-class ones
    apart, so projection-space disagreement tracks semantic rather than
    appearance mismatch. Computed on classes present in the batch.
    """
    B, C, h, w = pv.shape
    t = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode="nearest").long().squeeze(1)
    valid = (t != ignore_index)
    classes = [c for c in t.unique().tolist() if c != ignore_index and 0 <= c < num_classes]
    if len(classes) < 2:
        return pv.sum() * 0.0
    eps = 1e-6
    proto_v, proto_t = [], []
    for c in classes:
        m = ((t == c) & valid).unsqueeze(1).float()
        denom = m.sum() + eps
        proto_v.append((pv * m).sum((0, 2, 3)) / denom)
        proto_t.append((pt * m).sum((0, 2, 3)) / denom)
    Pv = F.normalize(torch.stack(proto_v), dim=1)
    Pt = F.normalize(torch.stack(proto_t), dim=1)
    sim = Pv @ Pt.t()  # cross-modal cosine, K x K
    K = sim.shape[0]
    eye = torch.eye(K, device=sim.device, dtype=torch.bool)
    pos = sim[eye].mean()  # same class across modalities
    neg = sim[~eye].mean() if (~eye).any() else torch.zeros((), device=sim.device)
    return (-pos + neg) * 0.5


def route_target_loss(router_logits, logit_v, logit_t, target,
                      route_beta=0.7, ignore_index=IGNORE_INDEX):
    """Direct router supervision from teacher correctness.

    Builds a per-pixel oracle of the cheapest correct path: V right -> V-trust,
    T right and V wrong -> T-rescue, both wrong -> Joint. Sending both-right
    pixels to the cheap V-trust path (not Joint) is what keeps the router from
    collapsing onto Joint and lets the rare T-rescue region actually form.
    The hard target is pooled to router resolution into a per-token distribution
    and matched with cross-entropy, weighted by each token's valid-pixel fraction.
    """
    h, w = router_logits.shape[-2:]
    with torch.no_grad():
        valid = (target != ignore_index)
        av = (logit_v.argmax(1) == target) & valid
        at = (logit_t.argmax(1) == target) & valid
        v_trust = (valid & av).float()  # V right (incl. both-right)
        t_rescue = (valid & at & ~av).float()  # T right, V wrong
        joint = (valid & ~av & ~at).float()  # both wrong
        tgt = torch.stack([v_trust, t_rescue, joint], 1)
        # T-rescue is rare (~3% of pixels), so weight each path by 1/frequency
        # (tempered by route_beta) to stop the majority paths swamping the CE.
        counts = tgt.sum((0, 2, 3))
        invf = counts.sum().clamp_min(1.0) / (3.0 * counts.clamp_min(1.0))
        w_path = invf ** route_beta
        w_path = w_path / w_path.mean()
        tgt = F.adaptive_avg_pool2d(tgt, (h, w))
        wmask = tgt.sum(1, keepdim=True)  # valid fraction per token
        tgt = tgt / wmask.clamp_min(1e-6)
    # log_softmax in fp32 (router_logits may be half under AMP) for a stable CE.
    logp = F.log_softmax(router_logits.float(), dim=1)
    ce = -(w_path.view(1, 3, 1, 1) * tgt * logp).sum(1, keepdim=True)
    return (ce * wmask).sum() / wmask.sum().clamp_min(1e-6)


def path_distill_loss(student_logits, teacher_logits, T=2.0):
    """Distil a single-modality teacher into the matching DiGToR pure path.

    The forced single-path output (`forward(force_path='v'|'t')`) is a shared
    under-capacity subnetwork, so it trails the standalone teacher. Distilling
    the teacher into it lifts the modality-failure fallback toward teacher
    quality. Standard KD: KL on temperature-softened logits scaled by T**2, run
    in fp32 since logits may be half under AMP."""
    s = F.log_softmax(student_logits.float() / T, dim=1)
    t = F.softmax(teacher_logits.float() / T, dim=1)
    kl = F.kl_div(s, t, reduction="none").sum(1)
    return kl.mean() * (T * T)


def path_cost_loss(pi, cost=(1.0, 1.0, 2.0)):
    """Compute-budget prior over routed paths.

    The Joint path runs both encoders plus a wide conv (~2x a single-modality
    path), so penalising path cost makes the router only choose Joint where the
    accuracy gain justifies it and otherwise fall back to a cheap single path.
    ``cost`` is per-path [V, T, Joint]."""
    c = pi.new_tensor(cost).view(1, len(cost), 1, 1)
    return (pi * c).mean()


def entropy_reg(pi):
    """Marginal-entropy regulariser over routed paths (anti-collapse)."""
    p_marg = pi.mean(dim=(0, 2, 3)).clamp_min(1e-6)
    H = -(p_marg * p_marg.log()).sum()
    return -H  # maximise marginal entropy
