import sys
import os
from pathlib import Path

import hydra
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import onnx
import onnxsim
import torch.onnx
from torch import nn

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.logging_utils import custom_logger


class YOLOv9WithPostprocess(nn.Module):
    def __init__(self, cfg: Config):
        super(YOLOv9WithPostprocess, self).__init__()
        self.device = torch.device(cfg.device)
        self.model = create_model(cfg.model, cfg.weight).to(self.device)
        self.vec2box = Vec2Box(self.model, cfg.image_size, self.device)

    def forward(self, tensor: torch.Tensor):
        raw_output = self.model(tensor)
        preds_cls, pred_anc, preds_box = self.vec2box(raw_output["Main"])
        return preds_cls, preds_box


@hydra.main(config_path="../yolo/config", config_name="config_m", version_base=None)
def main(cfg: Config):
    custom_logger()

    model_post = YOLOv9WithPostprocess(cfg)

    device = torch.device(cfg.device)
    input_shape = (1, 3, cfg.image_size[0], cfg.image_size[1])
    dummy_input = torch.randn(input_shape).to(device)

    model_name = f"{os.path.splitext(os.path.basename(cfg.weight))[0]}.vec2box.onnx"
    simp_model_name = os.path.splitext(model_name)[0] + ".sim.onnx"
    opset_version = 12
    torch.onnx.export(
        model_post,
        dummy_input,
        model_name,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["classes", "bbox_xyxy"],
    )

    onnx_model = onnx.load(model_name)
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, simp_model_name)


if __name__ == "__main__":
    main()
