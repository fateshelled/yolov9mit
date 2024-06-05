import sys
from pathlib import Path

import hydra
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import get_model
from yolo.utils.logging_utils import custom_logger
from yolo.utils.bounding_box_utils import AnchorBoxConverter
import torch.onnx


class YOLOv9WithPostprocess(torch.nn.Module):
    def __init__(self, cfg: Config):
        super(YOLOv9WithPostprocess, self).__init__()
        self.device = torch.device(cfg.device)
        self.model = get_model(cfg).to(self.device)
        self.anchor2box = AnchorBoxConverter(cfg, self.device)

    def forward(self, tensor: torch.Tensor):
        raw_output = self.model(tensor)
        predict, _ = self.anchor2box(raw_output[0][3:], with_logits=True)
        return predict


@hydra.main(config_path="../yolo/config", config_name="config", version_base=None)
def main(cfg: Config):
    custom_logger()

    device = torch.device(cfg.device)
    input_shape = (1, 3, cfg.image_size[0], cfg.image_size[1])
    dummy_input = torch.randn(input_shape).to(device)

    model_post = YOLOv9WithPostprocess(cfg)
    opset_version = 12
    torch.onnx.export(
        model_post,
        dummy_input,
        "yolov9mit_with_post.onnx",
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == "__main__":
    main()
