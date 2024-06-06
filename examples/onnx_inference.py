import time

import cv2
import numpy as np
import onnxruntime as ort

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

COLORS = [
    [0, 113, 188],
    [216, 82, 24],
    [236, 176, 31],
    [125, 46, 141],
    [118, 171, 47],
    [76, 189, 237],
    [161, 19, 46],
    [76, 76, 76],
    [153, 153, 153],
    [255, 0, 0],
    [255, 127, 0],
    [190, 190, 0],
    [0, 255, 0],
    [0, 0, 255],
    [170, 0, 255],
    [84, 84, 0],
    [84, 170, 0],
    [84, 255, 0],
    [170, 84, 0],
    [170, 170, 0],
    [170, 255, 0],
    [255, 84, 0],
    [255, 170, 0],
    [255, 255, 0],
    [0, 84, 127],
    [0, 170, 127],
    [0, 255, 127],
    [84, 0, 127],
    [84, 84, 127],
    [84, 170, 127],
    [84, 255, 127],
    [170, 0, 127],
    [170, 84, 127],
    [170, 170, 127],
    [170, 255, 127],
    [255, 0, 127],
    [255, 84, 127],
    [255, 170, 127],
    [255, 255, 127],
    [0, 84, 255],
    [0, 170, 255],
    [0, 255, 255],
    [84, 0, 255],
    [84, 84, 255],
    [84, 170, 255],
    [84, 255, 255],
    [170, 0, 255],
    [170, 84, 255],
    [170, 170, 255],
    [170, 255, 255],
    [255, 0, 255],
    [255, 84, 255],
    [255, 170, 255],
    [84, 0, 0],
    [127, 0, 0],
    [170, 0, 0],
    [212, 0, 0],
    [255, 0, 0],
    [0, 42, 0],
    [0, 84, 0],
    [0, 127, 0],
    [0, 170, 0],
    [0, 212, 0],
    [0, 255, 0],
    [0, 0, 42],
    [0, 0, 84],
    [0, 0, 127],
    [0, 0, 170],
    [0, 0, 212],
    [0, 0, 255],
    [0, 0, 0],
    [36, 36, 36],
    [72, 72, 72],
    [109, 109, 109],
    [145, 145, 145],
    [182, 182, 182],
    [218, 218, 218],
    [0, 113, 188],
    [80, 182, 188],
    [127, 127, 0],
]


def create_session(onnx_model_path: str, use_tensorrt=False) -> ort.InferenceSession:

    available_providers = ort.get_available_providers()
    providers = []
    if use_tensorrt and "TensorrtExecutionProvider" in available_providers:
        providers.append("TensorrtExecutionProvider")
    elif "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session_option = ort.SessionOptions()
    session_option.log_severity_level = 4
    session_option.intra_op_num_threads = 0

    onnx_session = ort.InferenceSession(
        onnx_model_path, sess_options=session_option, providers=providers
    )

    input = onnx_session.get_inputs()[0]
    output = onnx_session.get_outputs()[0]

    print("==== Model Info ====")
    print(f"Inputs:")
    print(f" - {input.name}, {input.shape}, {input.type}")
    print(f"Outputs:")
    print(f" - {output.name}, {output.shape}, {output.type}")
    print()
    return onnx_session


def get_input_shape(onnx_session: ort.InferenceSession):
    input = onnx_session.get_inputs()[0]
    input_height = input.shape[2]
    input_width = input.shape[3]
    return input_height, input_width


def preprocess(image_bgr: np.ndarray, input_shape: tuple[int, int]):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # resize and normalize
    blob = cv2.resize(image_rgb, input_shape[::-1]).astype(np.float32) / 255.0
    # HWC -> NCHW
    blob = blob.transpose(2, 0, 1)[None, :, :, :]
    return blob


def inference(onnx_session: ort.InferenceSession, blob: np.array):
    input = onnx_session.get_inputs()[0]
    input_name = input.name
    output = onnx_session.get_outputs()[0]
    output_name = output.name

    outputs = onnx_session.run([output_name], {input_name: blob})[0]
    return outputs


def calc_iou(bbox: np.ndarray, bboxes: np.ndarray, area: float, areas: np.ndarray):
    xA = np.maximum(bbox[0], bboxes[:, 0])
    yA = np.maximum(bbox[1], bboxes[:, 1])
    xB = np.minimum(bbox[2], bboxes[:, 2])
    yB = np.minimum(bbox[3], bboxes[:, 3])
    intersection = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    iou = intersection / (area + areas - intersection)
    return iou


def calc_nms(bboxes, cls_ids, confs, min_iou: float, class_num=80):
    new_bboxes = []
    new_cls_ids = []
    new_confs = []

    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

    sort_idx = np.lexsort((-confs, cls_ids))
    tmp_bboxes = bboxes[sort_idx]
    tmp_cls_ids = cls_ids[sort_idx]
    tmp_confs = confs[sort_idx]
    tmp_areas = areas[sort_idx]

    while len(tmp_bboxes) > 0:
        bbox = tmp_bboxes[0]
        cls_id = tmp_cls_ids[0]
        conf = tmp_confs[0]
        area = tmp_areas[0]
        new_bboxes.append(bbox)
        new_cls_ids.append(cls_id)
        new_confs.append(conf)

        other_bboxes = tmp_bboxes[1:]
        other_cls_ids = tmp_cls_ids[1:]
        other_confs = tmp_confs[1:]
        other_areas = tmp_areas[1:]

        iou = calc_iou(bbox, other_bboxes, area, other_areas)
        valid_mask = iou < min_iou
        tmp_bboxes = other_bboxes[valid_mask]
        tmp_cls_ids = other_cls_ids[valid_mask]
        tmp_confs = other_confs[valid_mask]
        tmp_areas = other_areas[valid_mask]

    return new_bboxes, new_cls_ids, new_confs


def postprocess(
    outputs: np.ndarray, min_confidence: float, min_iou: float, class_num=80
):
    cls_dist = outputs[:, :, :class_num]
    bbox = outputs[:, :, class_num:]

    cls_val = cls_dist.max(axis=-1, keepdims=True)
    cls_idx = np.argmax(cls_dist, axis=-1, keepdims=True)
    valid_mask = cls_val > min_confidence
    valid_cls = cls_idx[valid_mask]
    valid_con = cls_val[valid_mask]

    bbox_mask = np.repeat(valid_mask, 4, axis=2)
    valid_box = bbox[bbox_mask].reshape(-1, 4)

    return calc_nms(valid_box, valid_cls, valid_con, min_iou)


def draw_bboxes(
    image_bgr: np.ndarray,
    bboxes: np.ndarray,
    cls_ids: np.ndarray,
    confs: np.ndarray,
    input_shape: tuple[int, int],
):
    # bboxes: (DETECT_NUM, 4)
    # bbox: xyxy
    org_h, org_w = image_bgr.shape[:2]
    input_h, input_w = input_shape
    h_scale = org_h / input_h
    w_scale = org_w / input_w
    drawn = image_bgr.copy()
    for bbox, cls_id, conf in zip(bboxes, cls_ids, confs):
        x0, y0, x1, y1 = bbox.tolist()
        x0 = int(x0 * w_scale + 0.5)
        x1 = int(x1 * w_scale + 0.5)
        y0 = int(y0 * h_scale + 0.5)
        y1 = int(y1 * h_scale + 0.5)
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= org_w:
            x1 = org_w - 1
        if y1 >= org_h:
            y1 = org_h - 1
        cv2.rectangle(drawn, (x0, y0), (x1, y1), COLORS[cls_id%80], 1)
        cv2.putText(drawn, f"{COCO_NAMES[cls_id]}: {conf:.3f}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return drawn


def do_process(
    onnx_session: ort.InferenceSession,
    image_bgr: np.ndarray,
    input_shape: tuple[int, int],
    min_confidence: float,
    min_iou: float,
):

    t_pre = time.perf_counter()
    blob = preprocess(image_bgr, input_shape)
    dt_pre = time.perf_counter() - t_pre

    t_inf = time.perf_counter()
    outputs = inference(onnx_session, blob)
    dt_inf = time.perf_counter() - t_inf

    t_post = time.perf_counter()
    boxes, cls_ids, confs = postprocess(outputs, min_confidence, min_iou)
    dt_post = time.perf_counter() - t_post

    t_draw = time.perf_counter()
    drawn = draw_bboxes(image_bgr, boxes, cls_ids, confs, input_shape)
    dt_draw = time.perf_counter() - t_draw

    times = {
        "preprocess": dt_pre,
        "inference": dt_inf,
        "postprocess": dt_post,
        "draw": dt_draw,
    }

    return drawn, times


if __name__ == "__main__":

    image_path = "demo/images/inference/image.png"
    onnx_model_path = "yolov9mit_with_post.sim.onnx"
    use_tensorrt = False

    min_confidence = 0.5
    min_iou = 0.5

    onnx_session = create_session(onnx_model_path, use_tensorrt)
    input_shape = get_input_shape(onnx_session)

    image_bgr = cv2.imread(image_path)

    drawn, times = do_process(onnx_session, image_bgr, input_shape, min_confidence, min_iou)

    for i in range(5):
        drawn, times = do_process(onnx_session, image_bgr, input_shape, min_confidence, min_iou)
        print(f"--- {i} ---------------")
        for key, dt in times.items():
            tmp = f"{dt*1000:.3f}"
            print(f" {key.rjust(11)} {tmp.rjust(6)} millisec")

    cv2.imwrite("output.png", drawn)

    print()
    print("done.")
