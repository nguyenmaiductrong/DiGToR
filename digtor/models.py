import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NUM_CLASSES


def conv_bn_relu(ci, co, k=3, p=1):
    return nn.Sequential(nn.Conv2d(ci, co, k, padding=p, bias=False),
                         nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(conv_bn_relu(ci, co), conv_bn_relu(co, co))

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ci, co))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, ci, skip_c, co):
        super().__init__()
        self.up = nn.ConvTranspose2d(ci, ci // 2, 2, stride=2)
        self.conv = DoubleConv(ci // 2 + skip_c, co)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class Encoder(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16)
        self.out_channels = [base, base * 2, base * 4, base * 8, base * 16]

    def forward(self, x):
        s1 = self.inc(x)
        s2 = self.d1(s1)
        s3 = self.d2(s2)
        s4 = self.d3(s3)
        b = self.d4(s4)
        return [s1, s2, s3, s4, b]


class Decoder(nn.Module):
    def __init__(self, skip_chs, base=32, num_classes=NUM_CLASSES):
        super().__init__()
        self.u1 = Up(base * 16, skip_chs[3], base * 8)
        self.u2 = Up(base * 8, skip_chs[2], base * 4)
        self.u3 = Up(base * 4, skip_chs[1], base * 2)
        self.u4 = Up(base * 2, skip_chs[0], base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, feats):
        s1, s2, s3, s4, b = feats
        x = self.u1(b, s4)
        x = self.u2(x, s3)
        x = self.u3(x, s2)
        x = self.u4(x, s1)
        return self.out(x)


class UNet(nn.Module):
    """Single-stream multi-class UNet (V-only or T-only teacher)."""

    def __init__(self, in_ch=3, base=32, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc = Encoder(in_ch, base)
        self.dec = Decoder(self.enc.out_channels[:4], base, num_classes)

    def forward(self, x, return_feats=False):
        feats = self.enc(x)
        logit = self.dec(feats)
        if return_feats:
            return logit, feats
        return logit


class TwoStreamFusion(nn.Module):
    """Concat fusion at every skip + single decoder (uniform-fusion baseline)."""

    def __init__(self, base=32, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc_v = Encoder(3, base)
        self.enc_t = Encoder(1, base)
        chs = self.enc_v.out_channels
        self.reduce = nn.ModuleList([nn.Conv2d(2 * c, c, 1) for c in chs])
        self.dec = Decoder(chs[:4], base, num_classes)

    def forward(self, rgb, ir, return_feats=False):
        fv = self.enc_v(rgb)
        ft = self.enc_t(ir)
        feats = [r(torch.cat([a, b], 1)) for r, a, b in zip(self.reduce, fv, ft)]
        logit = self.dec(feats)
        if return_feats:
            return logit, fv, ft
        return logit


class ProjectionHead(nn.Module):
    """Task-aligned projection at the bottleneck (L2-normalised)."""

    def __init__(self, ci, co=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ci, co, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(co, co, 1))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ReliabilityHead(nn.Module):
    def __init__(self, ci):
        super().__init__()
        self.net = nn.Sequential(conv_bn_relu(ci, ci // 2), nn.Conv2d(ci // 2, 1, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class DiGToR(nn.Module):
    """Disagreement-Guided Token Router (multi-class).

    Routes each bottleneck token through Visible-trust / Thermal-rescue /
    Joint-fusion based on cross-modal disagreement (cosine distance in a
    task-aligned projection space) gated by per-modality reliability.
    """

    def __init__(self, base=32, proj_dim=64, tau=1.0, num_classes=NUM_CLASSES,
                 ablate_signals=None, ablate_routing=None, ablate_proj=None,
                 ablate_paths=3):
        super().__init__()
        self.enc_v = Encoder(3, base)
        self.enc_t = Encoder(1, base)
        chs = self.enc_v.out_channels
        Cb = chs[-1]
        # Per-skip projections into the decoder's skip space. The router drives
        # the whole decoder, so each skip needs a visible-only, thermal-only and
        # joint-fused variant in a common space.
        self.reduce = nn.ModuleList([nn.Conv2d(2 * c, c, 1) for c in chs[:4]])  # joint
        self.reduce_v = nn.ModuleList([nn.Conv2d(c, c, 1) for c in chs[:4]])  # V-trust skip
        self.reduce_t = nn.ModuleList([nn.Conv2d(c, c, 1) for c in chs[:4]])  # T-rescue skip

        self.proj_v = ProjectionHead(Cb, proj_dim)
        self.proj_t = ProjectionHead(Cb, proj_dim)
        self.rel_v = ReliabilityHead(Cb)
        self.rel_t = ReliabilityHead(Cb)

        self.path_v = DoubleConv(Cb, Cb)
        self.path_t = DoubleConv(Cb, Cb)
        self.path_j = DoubleConv(2 * Cb, Cb)

        self.router = nn.Sequential(
            nn.Conv2d(5, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1))

        # Hierarchical routing refinement, coarse to fine. The bottleneck router
        # decides at /16 where disagreement is semantically meaningful, but a
        # coarse token is never a majority of the sparse thermal-rescue pixels, so
        # a hard argmax there can never pick T-rescue. Refining the routing logits
        # up each skip scale with that scale's own reliability/disagreement lets a
        # sparse rescue region form a local majority and win the hard vote at the
        # finest scale, which is where L_route is supervised.
        self.proj_v_s = nn.ModuleList([nn.Conv2d(c, proj_dim, 1) for c in chs[:4]])
        self.proj_t_s = nn.ModuleList([nn.Conv2d(c, proj_dim, 1) for c in chs[:4]])
        self.rel_v_s = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in chs[:4]])
        self.rel_t_s = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in chs[:4]])
        self.refine = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 + 5, 16, 1), nn.ReLU(inplace=True),
                          nn.Conv2d(16, 3, 1)) for _ in chs[:4]])

        self.dec = Decoder(chs[:4], base, num_classes)
        self.tau = tau
        # Module ablations (defaults = full model). Each disables ONE architectural
        # component so its contribution can be measured by retraining:
        #   ablate_signals="d_only"          -> router sees disagreement only (zero c_V,c_T)
        #   ablate_routing="bottleneck_only" -> drop the coarse->fine skip refinement
        #   ablate_proj="raw"                -> disagreement from raw cosine (no projection)
        #   ablate_paths=2                   -> 2-path router (Joint-fusion path forbidden)
        self.ablate_signals = ablate_signals
        self.ablate_routing = ablate_routing
        self.ablate_proj = ablate_proj
        self.ablate_paths = ablate_paths
        # Learnable reliability-gate strength lambda (one global scalar). The gate
        # adds lambda*log(reliability) to the routing logits so the router will
        # not pick a modality it does not trust. Keeping lambda a parameter lets
        # each dataset learn its own value end-to-end under corrupt_aug. It goes
        # through softplus to stay non-negative; raw=0.5413 gives lambda~=1.0 at
        # init.
        self.rel_gate_raw = nn.Parameter(torch.tensor(0.5413))

    @property
    def rel_gate_lambda(self):
        return F.softplus(self.rel_gate_raw)

    def _route(self, logits, hard, tau):
        """Logits -> routing weights pi (one-hot if hard, else Gumbel/softmax)."""
        if hard:
            idx = logits.argmax(1, keepdim=True)
            return torch.zeros_like(logits).scatter_(1, idx, 1.0)
        t = self.tau if tau is None else tau
        if self.training:
            g = -torch.empty_like(logits).exponential_().log()
            return F.softmax((logits + g) / t, dim=1)
        return F.softmax(logits / t, dim=1)

    def routing_signals(self, fv_b, ft_b):
        if self.ablate_proj == "raw":
            # ablation: skip the task-aligned projection, use raw-feature cosine
            pv = F.normalize(fv_b, p=2, dim=1)
            pt = F.normalize(ft_b, p=2, dim=1)
        else:
            pv = self.proj_v(fv_b)
            pt = self.proj_t(ft_b)
        d = 1.0 - (pv * pt).sum(1, keepdim=True)  # cosine distance
        cv = self.rel_v(fv_b)
        ct = self.rel_t(ft_b)
        return d, cv, ct

    def _mask_joint(self, logits):
        """2-path ablation: forbid the Joint-fusion path so the router must choose
        between Visible-trust and Thermal-rescue only."""
        if self.ablate_paths == 2:
            logits = logits.clone()
            logits[:, 2:3] = -1e9
        return logits

    @staticmethod
    def _reliability_gate(logits, cv, ct, lam):
        """Add a reliability log-prior to the routing logits so the router will
        not select a modality it does not trust. Without it, reliability only
        gates the Joint path's features and the route choice itself ignores it,
        so a dead modality still leaks into the output. The gate is near-zero on
        clean inputs and bites hard under failure, recovering a single-modality
        fallback. `lam` is the gate strength; lam<=0 disables it.

        The Joint entry uses the geometric mean (mean of logs) of the two
        reliabilities, not the arithmetic mean: Joint needs both modalities, so it
        is only as reliable as the weakest one. The arithmetic mean barely
        penalises Joint when one modality dies, so tokens leak into it; the
        geometric mean penalises it as hard as the dead modality and routing
        collapses to the surviving one.
        """
        if isinstance(lam, (int, float)) and lam <= 0:
            return logits
        log_cv = cv.clamp_min(1e-4).log()
        log_ct = ct.clamp_min(1e-4).log()
        gate = torch.cat([log_cv, log_ct, 0.5 * (log_cv + log_ct)], 1)
        return logits + lam * gate

    def forward(self, rgb, ir, hard=False, return_aux=False, tau=None, rel_gate=None,
                force_path=None):
        # rel_gate: None uses the learnable lambda (default); a float overrides it
        # (0.0 ablates the gate).
        #
        # force_path in {None,'v','t'} bypasses the router and runs a single pure
        # path. force_path='v' runs enc_v + reduce_v + path_v + dec and never
        # touches the thermal encoder. It is both the distillation target matched
        # to the v_only teacher and the genuine hard skip used at inference when
        # thermal has failed: enc_t is skipped (real compute saved) with zero
        # thermal leak. 't' is symmetric.
        lam = self.rel_gate_lambda if rel_gate is None else rel_gate
        fv = self.enc_v(rgb)
        if force_path == "v":
            routed_b = self.path_v(fv[-1])
            skips = [self.reduce_v[i](fv[i]) for i in range(4)]
            logit = self.dec(skips + [routed_b])
            return (logit, dict(forced="v")) if return_aux else logit
        ft = self.enc_t(ir)
        if force_path == "t":
            routed_b = self.path_t(ft[-1])
            skips = [self.reduce_t[i](ft[i]) for i in range(4)]
            logit = self.dec(skips + [routed_b])
            return (logit, dict(forced="t")) if return_aux else logit
        fv_b, ft_b = fv[-1], ft[-1]

        d, cv, ct = self.routing_signals(fv_b, ft_b)
        if self.ablate_signals == "d_only":
            z = torch.zeros_like(d)
            sig = torch.cat([d, z, z, z, z], 1)   # router sees disagreement only
        else:
            sig = torch.cat([d, cv, ct, cv * d, ct * d], 1)
        logits = self.router(sig)  # B,3,h,w coarse (bottleneck) routing
        logits = self._reliability_gate(logits, cv, ct, lam)
        logits = self._mask_joint(logits)
        pi_b = self._route(logits, hard, tau)

        # Bottleneck routing. Gate the Joint path's modalities by their
        # reliability so Joint stops structurally dominating T-rescue: where
        # visible is unreliable (cv low) Joint loses its visible advantage and
        # collapses toward thermal, so picking the cheaper T-rescue path costs
        # nothing.
        out_v = self.path_v(fv_b)
        out_t = self.path_t(ft_b)
        out_j = self.path_j(torch.cat([cv * fv_b, ct * ft_b], 1))
        routed_b = pi_b[:, 0:1] * out_v + pi_b[:, 1:2] * out_t + pi_b[:, 2:3] * out_j

        # Hierarchical skip routing: the coarse decision is refined up each skip
        # scale (s4 to s1) with that scale's own disagreement/reliability. The
        # route drives the whole decoder, and at the finest scale the sparse
        # T-rescue regions form a local majority that can win the hard argmax.
        skips = [None] * 4
        cv_s = [None] * 4
        ct_s = [None] * 4
        cur = logits
        fine_logits, pi_fine = logits, pi_b
        for i in range(3, -1, -1):
            a, b = fv[i], ft[i]
            cvi = torch.sigmoid(self.rel_v_s[i](a))
            cti = torch.sigmoid(self.rel_t_s[i](b))
            cv_s[i], ct_s[i] = cvi, cti
            if self.ablate_routing == "bottleneck_only":
                # ablation: no per-scale refinement -- reuse the coarse decision
                # upsampled to this skip scale (the refine/proj_s modules are unused).
                lg = F.interpolate(logits, size=a.shape[-2:], mode="bilinear",
                                   align_corners=False)
            else:
                pv = F.normalize(self.proj_v_s[i](a), dim=1)
                pt = F.normalize(self.proj_t_s[i](b), dim=1)
                di = 1.0 - (pv * pt).sum(1, keepdim=True)
                up = F.interpolate(cur, size=a.shape[-2:], mode="bilinear",
                                   align_corners=False)
                # Not a residual on `up`: the coarse router collapses to V, and an
                # additive residual would impose that V-bias on every fine pixel. Feed
                # `up` as a feature only and let the supervised local reliability
                # decide, so the fine router can output T where visible is unreliable.
                lg = self.refine[i](torch.cat([up, di, cvi, cti,
                                               cvi * di, cti * di], 1))
                lg = self._reliability_gate(lg, cvi, cti, lam)
            lg = self._mask_joint(lg)
            cur = lg
            pi_i = self._route(lg, hard, tau)
            skips[i] = (pi_i[:, 0:1] * self.reduce_v[i](a)
                        + pi_i[:, 1:2] * self.reduce_t[i](b)
                        + pi_i[:, 2:3] * self.reduce[i](torch.cat([a, b], 1)))
            if i == 0:
                fine_logits, pi_fine = lg, pi_i
        logit = self.dec(skips + [routed_b])

        if return_aux:
            # pi / router_logits are the finest-scale routing (supervised by
            # L_route and measured by the routing analyses). cv_s/ct_s are the
            # per-scale reliability maps, which must be supervised or the fine
            # router cannot localise the sparse rescue region. d/cv/ct stay at the
            # bottleneck for the reliability diagnostics.
            return logit, dict(d=d, cv=cv, ct=ct, pi=pi_fine,
                               router_logits=fine_logits,
                               cv_s=cv_s, ct_s=ct_s,
                               fv_b=fv_b, ft_b=ft_b)
        return logit


def build_model(name, base=32, num_classes=NUM_CLASSES):
    name = name.lower()
    if name == "v_only":
        return UNet(in_ch=3, base=base, num_classes=num_classes)
    if name == "t_only":
        return UNet(in_ch=1, base=base, num_classes=num_classes)
    if name == "fusion":
        return TwoStreamFusion(base=base, num_classes=num_classes)
    if name == "digtor":
        return DiGToR(base=base, num_classes=num_classes)
    raise ValueError(name)
