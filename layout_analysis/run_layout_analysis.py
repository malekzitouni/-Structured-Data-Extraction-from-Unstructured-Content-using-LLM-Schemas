# run_layout_analysis.py
# This script runs layout analysis on a document image using the LayoutPredictor class.
# It requires the LayoutPredictor to be properly set up with the necessary model files.
import json
import logging
import os
import threading
from collections.abc import Iterable
from typing import Dict, List, Set, Union, Any, Tuple
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()

class LayoutPredictor:
    """
    Document layout prediction using safe tensors
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Set[str] = set(),
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        device: (Optional) device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'

        Raises
        ------
        FileNotFoundError when the model's torch file is missing
        """
        # Blacklisted classes
        self._black_classes = blacklist_classes

        # Canonical classes
        self._labels = LayoutLabels()

        # Set basic params
        self._threshold = base_threshold

        # Set number of threads for CPU
        self._device = torch.device(device)
        self._num_threads = num_threads
        if device == "cpu":
            torch.set_num_threads(self._num_threads)

        # Load model file and configurations
        self._processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        self._model_config = os.path.join(artifact_path, "config.json")
        self._st_fn = os.path.join(artifact_path, "model.safetensors")
        if not os.path.isfile(self._st_fn):
            raise FileNotFoundError("Missing safe tensors file: {}".format(self._st_fn))
        if not os.path.isfile(self._processor_config):
            raise FileNotFoundError(
                f"Missing processor config file: {self._processor_config}"
            )
        if not os.path.isfile(self._model_config):
            raise FileNotFoundError(f"Missing model config file: {self._model_config}")

        # Load model and move to device
        self._image_processor = RTDetrImageProcessor.from_json_file(
            self._processor_config
        )

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            self._model = AutoModelForObjectDetection.from_pretrained(
                artifact_path, config=self._model_config
            ).to(self._device)
            self._model.eval()

        # Set classes map
        self._model_name = type(self._model).__name__
        if self._model_name == "RTDetrForObjectDetection":
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0

        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "model_name": self._model_name,
            "safe_tensors_file": self._st_fn,
            "device": self._device.type,
            "num_threads": self._num_threads,
            "image_size": self._image_processor.size,
            "threshold": self._threshold,
        }
        return info

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence", "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        target_sizes = torch.tensor([page_img.size[::-1]])
        inputs = self._image_processor(images=[page_img], return_tensors="pt").to(
            self._device
        )
        outputs = self._model(**inputs)
        results: List[Dict[str, Tensor]] = (
            self._image_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self._threshold,
            )
        )

        w, h = page_img.size
        result = results[0]
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score = float(score.item())

            label_id = int(label_id.item()) + self._label_offset
            label_str = self._classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self._black_classes:
                continue

            bbox_float = [float(b.item()) for b in box]
            l = min(w, max(0, bbox_float[0]))
            t = min(h, max(0, bbox_float[1]))
            r = min(w, max(0, bbox_float[2]))
            b = min(h, max(0, bbox_float[3]))
            yield {
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "label": label_str,
                "confidence": score,
            }

    @torch.inference_mode()
    def predict_batch(
        self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """
        Batch prediction for multiple images - more efficient than calling predict() multiple times.

        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to process in a single batch

        Returns
        -------
        List[List[dict]]
            List of prediction lists, one per input image. Each prediction dict contains:
            "label", "confidence", "l", "t", "r", "b"
        """
        if not images:
            return []

        # Convert all images to RGB PIL format
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Not supported input image format")

        # Get target sizes for all images
        target_sizes = torch.tensor([img.size[::-1] for img in pil_images])

        # Process all images in a single batch
        inputs = self._image_processor(images=pil_images, return_tensors="pt").to(
            self._device
        )
        outputs = self._model(**inputs)

        # Post-process all results at once
        results_list: List[Dict[str, Tensor]] = (
            self._image_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self._threshold,
            )
        )

        # Convert results to standard format for each image
        all_predictions = []

        for img, results in zip(pil_images, results_list):
            w, h = img.size
            predictions = []

            for score, label_id, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                score = float(score.item())
                label_id = int(label_id.item()) + self._label_offset
                label_str = self._classes_map[label_id]

                # Filter out blacklisted classes
                if label_str in self._black_classes:
                    continue

                bbox_float = [float(b.item()) for b in box]
                l = min(w, max(0, bbox_float[0]))
                t = min(h, max(0, bbox_float[1]))
                r = min(w, max(0, bbox_float[2]))
                b = min(h, max(0, bbox_float[3]))

                predictions.append(
                    {
                        "l": l,
                        "t": t,
                        "r": r,
                        "b": b,
                        "label": label_str,
                        "confidence": score,
                    }
                )

            all_predictions.append(predictions)

        return all_predictions
    

class LayoutLabels:
    r"""Single point of reference for the layout labels"""

    def __init__(self) -> None:
        r""" """
        # Canonical classes originating in DLNv2
        self._canonical: Dict[int, str] = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
            11: "Document Index",
            12: "Code",
            13: "Checkbox-Selected",
            14: "Checkbox-Unselected",
            15: "Form",
            16: "Key-Value Region",
        }
        self._inverse_canonical: Dict[str, int] = {
            label: class_id for class_id, label in self._canonical.items()
        }

        # Shifted canonical classes with background in 0
        self._shifted_canonical: Dict[int, str] = {0: "Background"}
        for k, v in self._canonical.items():
            self._shifted_canonical[k + 1] = v
        self._inverse_shifted_canonical: Dict[str, int] = {
            label: class_id for class_id, label in self._shifted_canonical.items()
        }

    def canonical_categories(self) -> Dict[int, str]:
        return self._canonical

    def canonical_to_int(self) -> Dict[str, int]:
        return self._inverse_canonical

    def shifted_canonical_categories(self) -> Dict[int, str]:
        return self._shifted_canonical

    def shifted_canonical_to_int(self) -> Dict[str, int]:
        return self._inverse_shifted_canonical
    
def save_predictions(
    prefix: str, 
    viz_dir: str, 
    img_fn: Path, 
    img: Image.Image, 
    predictions: List[Dict[str, Any]]
):
    img_path = Path(img_fn)

    image = img.copy()
    draw = ImageDraw.Draw(image)

    predictions_filename = f"{prefix}_{img_path.stem}.txt"
    predictions_fn = os.path.join(viz_dir, predictions_filename)
    with open(predictions_fn, "w") as fd:
        for pred in predictions:
            bbox = [
                round(pred["l"], 2),
                round(pred["t"], 2),
                round(pred["r"], 2),
                round(pred["b"], 2),
            ]
            label = pred["label"]
            confidence = round(pred["confidence"], 3)

            # Save the predictions in txt file
            pred_txt = f"{prefix} {str(img_fn)}: {label} - {bbox} - {confidence}\n"
            fd.write(pred_txt)

            # Draw the bbox and label
            draw.rectangle(bbox, outline="orange")
            txt = f"{label}: {confidence}"
            draw.text(
                (bbox[0], bbox[1]), text=txt, font=ImageFont.load_default(), fill="blue"
            )

    # --- Save predictions as JSON ---
    predictions_json_filename = f"{prefix}_{img_path.stem}.json"
    predictions_json_fn = os.path.join(viz_dir, predictions_json_filename)
    with open(predictions_json_fn, "w", encoding="utf-8") as json_fd:
        json.dump(predictions, json_fd, indent=2, ensure_ascii=False)
    # --- End JSON saving ---

    draw_filename = f"{prefix}_{img_path.name}"
    draw_fn = os.path.join(viz_dir, draw_filename)
    image.save(draw_fn)
    return draw_fn


def process_image(
    logger: logging.Logger,
    artifact_path: str,
    device: str,
    num_threads: int,
    img_path: str,
    viz_dir: str,
    threshold: float,
):
    """
    Process a single image using LayoutPredictor
    """
    # Create the layout predictor
    predictor = LayoutPredictor(
        artifact_path, 
        device=device, 
        num_threads=num_threads, 
        base_threshold=threshold
    )

    # Process the single image
    logger.info("Processing image: '%s'", img_path)
    img_path = Path(img_path)
    if not img_path.exists():
        logger.error("Image file not found: %s", img_path)
        return

    try:
        with Image.open(img_path) as image:
            # Predict layout
            img_t0 = time.perf_counter()
            preds: List[Dict[str, Any]] = list(predictor.predict(image))
            img_ms = 1000 * (time.perf_counter() - img_t0)
            logger.info("Prediction time: %.2f ms", img_ms)

            # Print predictions to console
            print(f"\n{'='*50}")
            print(f"Layout predictions for: {img_path}")
            print(f"{'Label':<20} {'Confidence':<10} {'Bounding Box':<40}")
            print('-'*70)
            for i, pred in enumerate(preds):
                bbox = [
                    round(pred["l"], 2),
                    round(pred["t"], 2),
                    round(pred["r"], 2),
                    round(pred["b"], 2)
                ]
                print(f"{i+1:>2}. {pred['label']:<16} {pred['confidence']:<10.3f} {str(bbox):<40}")
            print('-'*70)
            print(f"Total predictions: {len(preds)}")
            print(f"{'='*50}\n")

            # Save predictions and visualization
            logger.info("Saving prediction visualization in: '%s'", viz_dir)
            viz_path = save_predictions("ST", viz_dir, img_path, image, preds)
            logger.info("Visualization saved to: %s", viz_path)
            
            return preds
            
    except Exception as e:
        logger.error("Error processing image: %s", e)
        return None


def main(args):
    r""" """
    num_threads = int(args.num_threads) if args.num_threads is not None else 4
    device = args.device.lower()
    img_path = args.img_path
    viz_dir = args.viz_dir
    hugging_face_repo = args.hugging_face_repo
    threshold = float(args.threshold)

    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LayoutPredictor")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure the viz dir exists
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Download models from HF
    logger.info("Downloading model from Hugging Face Hub: %s", hugging_face_repo)
    download_path = snapshot_download(repo_id=hugging_face_repo)

    # Process the image
    process_image(
        logger=logger,
        artifact_path=download_path,
        device=device,
        num_threads=num_threads,
        img_path=img_path,
        viz_dir=viz_dir,
        threshold=threshold
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Layout Prediction")

    supported_hf_repos = [
        "ds4sd/docling-layout-old",
        "ds4sd/docling-layout-heron",
        "ds4sd/docling-layout-heron-101",
        "ds4sd/docling-layout-egret-medium",
        "ds4sd/docling-layout-egret-large",
        "ds4sd/docling-layout-egret-xlarge",
    ]
    parser.add_argument(
        "-r",
        "--hugging-face-repo",
        required=False,
        default="ds4sd/docling-layout-old",
        help=f"The hugging face repo id: [{', '.join(supported_hf_repos)}]",
    )
    parser.add_argument(
        "-t", 
        "--threshold", 
        required=False, 
        default=0.3, 
        help="Confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "-d", 
        "--device", 
        required=False, 
        default="cpu", 
        help="One of [cpu, cuda, mps]"
    )
    parser.add_argument(
        "-n", 
        "--num_threads", 
        required=False, 
        default=4, 
        help="Number of threads (CPU only)"
    )
    parser.add_argument(
        "-i",
        "--img_path",
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument(
        "-v",
        "--viz_dir",
        required=False,
        default="viz_output",
        help="Directory to save prediction visualizations",
    )

    args = parser.parse_args()
    
    # Convert Windows path if needed
    if sys.platform == "win32":
        args.img_path = os.path.normpath(args.img_path)
    
    main(args)