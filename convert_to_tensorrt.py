from ultralytics import YOLO
import subprocess
import os
import onnx
import numpy as np
import tensorrt as trt
import argparse

def onnx_to_tensorrt(onnx_path, engine_file):
    """
    Converts onnx to engine file as the optimized model for usage.
    """
    onnx_model = onnx.load(onnx_path)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not parser.parse(onnx_model.SerializeToString()):
        error_count = parser.num_errors
        for error_idx in range(error_count):
            error = parser.get_error(error_idx)
            print(f"[ERROR] {error.code()}: {error.desc()}")
        raise RuntimeError(f"Failed to parse ONNX model with {error_count} errors")
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  
    
    config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)

    if engine is not None:
        with open(engine_file, "wb") as f:
            f.write(engine.serialize())
    else:
        print("[ERROR] Failed to build engine")


def convert_to_trt(models_path, inference_method, imgsz=416, device='cuda'):
    """
    Prunes and converts a YOLO .pt model to a TensorRT .engine file.
    """
    pt_path = os.path.join(models_path, inference_method, 'best.pt')
    onnx_path = os.path.join(models_path, inference_method, 'best.onnx')
    slimmed_path = os.path.join(models_path, inference_method, 'best.slim.onnx')
    engine_path = os.path.join(models_path, inference_method, 'best.engine')

    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Model file not found: {pt_path}")

    print(f"Loading model: {pt_path}")
    model = YOLO(pt_path)

    print(f"Exporting to ONNX...")
    model.export(format='onnx', imgsz=imgsz)

    print(f"Pruning ONNX model...")
    subprocess.run([
        "onnxslim", onnx_path, slimmed_path, '--dtype', 'fp16'
    ], check=True)

    #model = YOLO(onnx_path)
    #task = "detect"
    #if(inference_method == "segmentation"):
    #    task = 'segment'
    #model.export(format='engine', imgsz=imgsz)
    onnx_to_tensorrt(onnx_path, engine_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert YOLOv8 .pt model to TensorRT .engine file")
    parser.add_argument("--models_path", default="models" , help="Path to the .pt file")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size (default: 416)")
    parser.add_argument("--inference_method", type=str, default="detection", help="detection or segmentation")

    args = parser.parse_args()

    convert_to_trt(args.models_path, args.inference_method, imgsz=args.imgsz)
