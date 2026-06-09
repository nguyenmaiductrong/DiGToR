from dataclasses import dataclass
from importlib import import_module


IGNORE_INDEX = 255

FMB_CLASS_NAMES = [
    "unlabelled", "road", "sidewalk", "building", "lamp", "sign",
    "vegetation", "sky", "person", "car", "truck", "bus",
    "motorcycle", "bicycle", "pole",
]

SEMRT_CLASS_NAMES = [
    "unlabelled", "car_stop", "bike", "bicyclist", "motorcycle", "motorcyclist",
    "car", "tricycle", "traffic_light", "box", "pole", "curve", "person",
]

SEMRT_PALETTE = {
    0: (0, 0, 0),
    1: (72, 61, 39),
    2: (0, 0, 255),
    3: (148, 0, 211),
    4: (128, 128, 0),
    5: (64, 64, 128),
    6: (0, 139, 139),
    7: (131, 139, 139),
    8: (192, 64, 0),
    9: (126, 192, 238),
    10: (244, 164, 96),
    11: (211, 211, 211),
    12: (205, 155, 155),
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    log_prefix: str
    num_classes: int
    class_names: list[str]
    default_ckpt_dir: str
    default_results_dir: str
    default_wandb_project: str
    root_help: str
    flops_source: str

    @property
    def detector_out(self):
        return f"{self.default_results_dir}/detector.json"

    @property
    def rescue_out(self):
        return f"{self.default_results_dir}/eval_rescue.json"

    @property
    def robustness_out(self):
        return f"{self.default_results_dir}/robustness.json"

    @property
    def modality_cut_out(self):
        return f"{self.default_results_dir}/modality_cut.json"

    @property
    def flops_out(self):
        return f"{self.default_results_dir}/flops.json"


DATASET_CONFIGS = {
    "fmb": DatasetConfig(
        name="fmb",
        display_name="FMB",
        log_prefix="fmb",
        num_classes=15,
        class_names=FMB_CLASS_NAMES,
        default_ckpt_dir="ckpt_fmb",
        default_results_dir="results_fmb",
        default_wandb_project="digtor-fmb",
        root_help="FMB root with Visible/Infrared/Label",
        flops_source="MEASURED on FMB test",
    ),
    "semanticrt": DatasetConfig(
        name="semanticrt",
        display_name="SemanticRT",
        log_prefix="semrt",
        num_classes=13,
        class_names=SEMRT_CLASS_NAMES,
        default_ckpt_dir="ckpt_semrt",
        default_results_dir="results_semrt",
        default_wandb_project="digtor-semanticrt",
        root_help="SemanticRT root with rgb/thermal/labels + split txt files",
        flops_source="MEASURED on SemanticRT test",
    ),
}

# Backwards-compatible defaults for code that imports digtor.NUM_CLASSES directly.
NUM_CLASSES = DATASET_CONFIGS["fmb"].num_classes


def dataset_choices():
    return tuple(DATASET_CONFIGS)


def get_dataset_config(name):
    try:
        return DATASET_CONFIGS[name.lower()]
    except KeyError as e:
        raise ValueError(f"unknown dataset {name!r}; choose from {dataset_choices()}") from e


def get_dataset_module(name):
    cfg = get_dataset_config(name)
    return import_module(f".dataset.{cfg.name}", __name__)


def enable_fast_gpu():
    # TF32 matmul + cuDNN autotuning. Safe here because the input size is fixed,
    # and a big speedup on A100-class GPUs with no meaningful accuracy change.
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
