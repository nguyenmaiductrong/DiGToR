"""Single source of truth for the DiGToR demo.

Every number here is transcribed verbatim from the evaluation logs
(`digtor-fmb.log`, `digtor-semanticrt.log`) so the Streamlit app and the paper
report exactly the same figures. Do not hand-edit; regenerate from the logs.
"""

PATHS = ["V-trust", "T-rescue", "Joint"]
REGIONS = ["t_rescue", "v_preserve", "easy", "hard"]

DATA = {
    "FMB": {
        "title": "FMB (ICCV 2023) — dev rig, 14 lop, ~1.5k cap RGB-hong ngoai",
        # region -> % of valid pixels
        "regions_pct": {"t_rescue": 5.70, "v_preserve": 7.32, "easy": 79.63, "hard": 7.35},
        # detector AUROC/AUPRC for the thermal-rescue detection task
        "detector": {
            "low_conf_v":          {"auroc": 0.8766, "auprc": 0.2455, "kind": "baseline"},
            "ent_v":               {"auroc": 0.8753, "auprc": 0.2354, "kind": "baseline"},
            "dark_v":              {"auroc": 0.5640, "auprc": 0.0782, "kind": "baseline"},
            "d_prob":              {"auroc": 0.9407, "auprc": 0.3530, "kind": "digtor"},
            "LR(d,c_v,c_t)":       {"auroc": 0.9517, "auprc": 0.3995, "kind": "digtor"},
        },
        "decision_gate": {"best_base": (0.8766, 0.2455), "best_digtor": (0.9517, 0.3995),
                          "verdict": "PASS"},
        # model -> (mIoU, mAcc, FWIoU) clean
        "segmentation": {
            "v_only": (0.4738, 0.5530, 0.7739),
            "t_only": (0.4299, 0.5106, 0.7520),
            "fusion": (0.5066, 0.5839, 0.8021),
            "digtor": (0.5024, 0.5699, 0.7963),
        },
        # region -> {path: pct routed}
        "routing_alignment": {
            "t_rescue":   {"V-trust": 50.50, "T-rescue": 20.57, "Joint": 28.93},
            "v_preserve": {"V-trust": 62.06, "T-rescue": 7.49,  "Joint": 30.44},
            "easy":       {"V-trust": 90.94, "T-rescue": 2.25,  "Joint": 6.82},
            "hard":       {"V-trust": 46.76, "T-rescue": 11.87, "Joint": 41.38},
        },
        # condition -> {paths..., n, miou}
        "conditions": {
            "daylight": {"n": 78,  "V-trust": 85.56, "T-rescue": 1.90, "Joint": 12.54, "miou": 0.4742},
            "fog":      {"n": 81,  "V-trust": 83.24, "T-rescue": 4.33, "Joint": 12.44, "miou": 0.5475},
            "lowlight": {"n": 12,  "V-trust": 80.44, "T-rescue": 6.84, "Joint": 12.72, "miou": 0.3709},
            "other":    {"n": 109, "V-trust": 81.98, "T-rescue": 5.93, "Joint": 12.09, "miou": 0.4841},
        },
        "day_night": {"day_trescue": 2.37, "night_trescue": 6.40, "diff": 0.0403,
                      "p": 0.0000, "significant": True, "median_lum": 0.487},
        # tag -> {model: mIoU}
        "robustness": {
            "clean":            {"v_only": 0.4738, "t_only": 0.4299, "fusion": 0.5066, "digtor": 0.5024},
            "th_noise_0.3":     {"v_only": 0.4738, "t_only": 0.0194, "fusion": 0.4818, "digtor": 0.4891},
            "th_noise_0.6":     {"v_only": 0.4738, "t_only": 0.0273, "fusion": 0.4662, "digtor": 0.4845},
            "th_crossover_0.5": {"v_only": 0.4738, "t_only": 0.3569, "fusion": 0.4930, "digtor": 0.4983},
            "th_crossover_0.8": {"v_only": 0.4738, "t_only": 0.2058, "fusion": 0.4700, "digtor": 0.4922},
            "th_low_contrast":  {"v_only": 0.4738, "t_only": 0.2703, "fusion": 0.4810, "digtor": 0.4959},
            "th_dropout":       {"v_only": 0.4738, "t_only": 0.0698, "fusion": 0.4404, "digtor": 0.4812},
            "v_darken_0.6":     {"v_only": 0.3496, "t_only": 0.4299, "fusion": 0.4726, "digtor": 0.4751},
            "v_noise_0.3":      {"v_only": 0.1780, "t_only": 0.4299, "fusion": 0.4908, "digtor": 0.4700},
            "v_blur":           {"v_only": 0.3573, "t_only": 0.4299, "fusion": 0.4853, "digtor": 0.4878},
            "v_fog_0.5":        {"v_only": 0.3593, "t_only": 0.4299, "fusion": 0.4785, "digtor": 0.4786},
        },
        # reliability signal means under corruption: tag -> (d, c_v, c_t)
        "signals_under_corruption": {
            "clean":            (0.2000, 0.9421, 0.9048),
            "th_noise_0.3":     (0.3923, 0.9421, 0.2089),
            "th_noise_0.6":     (0.5678, 0.9421, 0.1766),
            "th_crossover_0.8": (0.2788, 0.9421, 0.6296),
            "th_dropout":       (0.7550, 0.9421, 0.4775),
            "v_noise_0.3":      (0.3337, 0.4980, 0.9048),
            "v_blur":           (0.3066, 0.8502, 0.9048),
        },
        # model -> (thermal-MFR, visible-MFR)
        "mfr": {
            "fusion": (0.9318, 0.9510),
            "digtor": (0.9757, 0.9512),
        },
        # GFLOPs
        "flops": {
            "fusion": 102.255, "digtor_dense": 136.947,
            "token_routing": 114.068, "single_encoder_skip": 81.559,
        },
        "latency_ms": {"v_only": 6.76, "t_only": 6.85, "fusion": 10.83,
                       "digtor_dense": 18.98, "digtor_force_v": 7.76},
        "route_share": {"V-trust": 0.833, "T-rescue": 0.044, "Joint": 0.124},
        "modality_cut": {
            "verdict": "PASS",
            "clean_delta": -0.0043,
            "th_noise_0.6": +0.0192, "th_dropout": +0.0450,
            "note": "Cat modality khi cam bien chet VUOT fusion.",
        },
    },
    "SemanticRT": {
        "title": "SemanticRT (ACM MM 2023) — headline, ~11k cap, thien ve canh dem/thieu sang",
        "regions_pct": {"t_rescue": 4.02, "v_preserve": 1.82, "easy": 87.98, "hard": 6.18},
        "detector": {
            "low_conf_v":          {"auroc": 0.8236, "auprc": 0.2223, "kind": "baseline"},
            "ent_v":               {"auroc": 0.8266, "auprc": 0.2352, "kind": "baseline"},
            "dark_v":              {"auroc": 0.5317, "auprc": 0.0453, "kind": "baseline"},
            "d_prob":              {"auroc": 0.9804, "auprc": 0.6581, "kind": "digtor"},
            "LR(d,c_v,c_t)":       {"auroc": 0.9858, "auprc": 0.6179, "kind": "digtor"},
        },
        "decision_gate": {"best_base": (0.8266, 0.2352), "best_digtor": (0.9858, 0.6581),
                          "verdict": "PASS"},
        "segmentation": {
            "v_only": (0.7170, 0.8265, 0.8297),
            "t_only": (0.7794, 0.8767, 0.8576),
            "fusion": (0.7843, 0.8793, 0.8600),
            "digtor": (0.7675, 0.8608, 0.8536),
        },
        "routing_alignment": {
            "t_rescue":   {"V-trust": 20.19, "T-rescue": 37.12, "Joint": 42.69},
            "v_preserve": {"V-trust": 22.36, "T-rescue": 21.08, "Joint": 56.56},
            "easy":       {"V-trust": 76.02, "T-rescue": 3.49,  "Joint": 20.49},
            "hard":       {"V-trust": 25.33, "T-rescue": 10.16, "Joint": 64.51},
        },
        "conditions": {
            "daylight": {"n": 114,  "V-trust": 70.82, "T-rescue": 3.23, "Joint": 25.95, "miou": 0.6653},
            "lowlight": {"n": 2438, "V-trust": 69.42, "T-rescue": 5.91, "Joint": 24.66, "miou": 0.7706},
            "other":    {"n": 284,  "V-trust": 70.96, "T-rescue": 4.14, "Joint": 24.90, "miou": 0.7779},
        },
        "day_night": {"day_trescue": 5.42, "night_trescue": 5.76, "diff": 0.0017,
                      "p": 0.3590, "significant": False, "median_lum": 0.139},
        "robustness": {
            "clean":            {"v_only": 0.7170, "t_only": 0.7794, "fusion": 0.7843, "digtor": 0.7675},
            "th_noise_0.3":     {"v_only": 0.7170, "t_only": 0.0069, "fusion": 0.7631, "digtor": 0.7464},
            "th_noise_0.6":     {"v_only": 0.7170, "t_only": 0.0025, "fusion": 0.7427, "digtor": 0.7377},
            "th_crossover_0.5": {"v_only": 0.7170, "t_only": 0.7434, "fusion": 0.7821, "digtor": 0.7646},
            "th_crossover_0.8": {"v_only": 0.7170, "t_only": 0.5018, "fusion": 0.7757, "digtor": 0.7579},
            "th_low_contrast":  {"v_only": 0.7170, "t_only": 0.6485, "fusion": 0.7792, "digtor": 0.7612},
            "th_dropout":       {"v_only": 0.7170, "t_only": 0.0000, "fusion": 0.6868, "digtor": 0.7168},
            "v_darken_0.6":     {"v_only": 0.2148, "t_only": 0.7794, "fusion": 0.7815, "digtor": 0.7802},
            "v_noise_0.3":      {"v_only": 0.3121, "t_only": 0.7794, "fusion": 0.7809, "digtor": 0.7773},
            "v_blur":           {"v_only": 0.4618, "t_only": 0.7794, "fusion": 0.7796, "digtor": 0.7652},
            "v_fog_0.5":        {"v_only": 0.3812, "t_only": 0.7794, "fusion": 0.7819, "digtor": 0.7780},
        },
        "signals_under_corruption": {
            "clean":            (0.6875, 0.9716, 0.9739),
            "th_noise_0.3":     (0.7650, 0.9716, 0.7951),
            "th_noise_0.6":     (0.8077, 0.9716, 0.7882),
            "th_crossover_0.8": (0.7102, 0.9716, 0.9131),
            "th_dropout":       (1.1080, 0.9716, 0.7632),
            "v_noise_0.3":      (0.7115, 0.8443, 0.9739),
            "v_blur":           (0.8027, 0.8798, 0.9739),
        },
        "mfr": {
            "fusion": (0.9626, 0.9958),
            "digtor": (0.9739, 1.0101),
        },
        "flops": {
            "fusion": 102.230, "digtor_dense": 136.922,
            "token_routing": 113.805, "single_encoder_skip": 81.534,
        },
        "latency_ms": {"v_only": 3.81, "t_only": 3.55, "fusion": 5.09,
                       "digtor_dense": 11.86, "digtor_force_v": 3.90},
        "route_share": {"V-trust": 0.886, "T-rescue": 0.020, "Joint": 0.094},
        "modality_cut": {
            "verdict": "FAIL",
            "clean_delta": -0.0168,
            "th_noise_0.6": -0.0051, "th_dropout": +0.0300,
            "note": "Nhanh nhiet manh (t_only 0.779 > v_only 0.717) -> cat cung khong "
                    "thang fusion duoi cai chet nhiet. Luan diem 'cat modality' KHONG "
                    "khai quat -> bang chung cho viec pivot sang chan doan (C2) + suy giam duyen dang (C3).",
        },
    },
}

# Routing-path colours shared by the precompute script and the app (RGB 0-255).
PATH_COLORS = {
    "V-trust":  (66, 135, 245),   # blue  -> trust visible
    "T-rescue": (240, 90, 60),    # red   -> thermal rescue
    "Joint":    (120, 200, 90),   # green -> joint fusion
}
REGION_COLORS = {
    "t_rescue":   (240, 90, 60),
    "v_preserve": (66, 135, 245),
    "easy":       (200, 200, 200),
    "hard":       (250, 210, 60),
}
