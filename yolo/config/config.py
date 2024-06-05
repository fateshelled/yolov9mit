from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from torch import nn


@dataclass
class AnchorConfig:
    reg_max: int
    strides: List[int]


@dataclass
class LayerConfg:
    args: Dict
    source: Union[int, str, List[int]]
    tags: str


@dataclass
class BlockConfig:
    block: List[Dict[str, LayerConfg]]


@dataclass
class Model:
    anchor: AnchorConfig
    model: Dict[str, BlockConfig]


@dataclass
class DownloadDetail:
    url: str
    file_size: int


@dataclass
class DownloadOptions:
    details: Dict[str, DownloadDetail]


@dataclass
class DatasetConfig:
    path: str
    auto_download: Optional[DownloadOptions]


@dataclass
class DataConfig:
    shuffle: bool
    batch_size: int
    pin_memory: bool
    data_augment: Dict[str, int]


@dataclass
class OptimizerArgs:
    lr: float
    weight_decay: float


@dataclass
class OptimizerConfig:
    type: str
    args: OptimizerArgs


@dataclass
class MatcherConfig:
    iou: str
    topk: int
    factor: Dict[str, int]


@dataclass
class LossConfig:
    objective: Dict[str, int]
    aux: Union[bool, float]
    matcher: MatcherConfig


@dataclass
class SchedulerConfig:
    type: str
    warmup: Dict[str, Union[int, float]]
    args: Dict[str, Any]


@dataclass
class EMAConfig:
    enabled: bool
    decay: float


@dataclass
class TrainConfig:
    task: str
    dataset: DatasetConfig
    epoch: int
    data: DataConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    scheduler: SchedulerConfig
    ema: EMAConfig


@dataclass
class NMSConfig:
    min_confidence: float
    min_iou: float


@dataclass
class InferenceConfig:
    task: str
    source: Union[str, int]
    nms: NMSConfig


@dataclass
class Config:
    task: Union[TrainConfig, InferenceConfig]
    model: Model
    name: str

    device: Union[str, int, List[int]]
    cpu_num: int

    class_num: int
    image_size: List[int]

    out_path: str
    exist_ok: bool

    lucky_number: 10
    use_wandb: bool
    use_TensorBoard: bool

    weight: Optional[str]


@dataclass
class YOLOLayer(nn.Module):
    source: Union[int, str, List[int]]
    output: bool
    tags: str
    layer_type: str
    usable: bool

    def __post_init__(self):
        super().__init__()
