import os
import json
import csv
from PIL import Image

from src.loader import DataLoader
from src.preprocess import Preprocessor
from src.segmentation import ImageSegmentation
from src.building_type_enum import BuildingType
from src.vision_model import VisionModel


class BuildingPipeline:

    def __init__(self, data_dir, output_dir, logger):

        self.logger = logger

        self.data_dir = data_dir
        self.output_dir = output_dir

        self.loader = DataLoader(data_dir)
        self.preprocessor = Preprocessor()
        self.segmentation = ImageSegmentation()
        self.classifier = VisionModel()

        os.makedirs(output_dir, exist_ok=True)

        self.prediction_file = os.path.join(output_dir, "predictions.csv")
        self.intermediate_file = os.path.join(
            output_dir, "intermediate_results.jsonl"
        )

    def run(self):

        self.logger.info("Loading metadata")

        locations = self.loader.load_locations()

        final_results = {}

        for location in locations:
            location_id = location["location_id"]
            self.logger.info(f"Processing {location_id}")
            images = location["images"]
            final_results[location_id] = {}
            final_results[location_id]["year_wise_prediction"] = {}
            for img_info in images:
                year = img_info["year"]
                path = img_info["path"]

                image_result = {
                    "image_path": str(path)
                }
                image_result.update(self.classify_building(path))
                final_results[location_id]["year_wise_prediction"][year] = image_result
            final_prediction, scores = self.aggregate_predictions(
                final_results[location_id]["year_wise_prediction"])
            final_results[location_id]["final_prediction"] = final_prediction
            final_results[location_id]["scores"] = scores

        print(final_results)

    def classify_building(self, path: str) -> dict:
        try:
            image_result = {}
            image = Image.open(path).convert("RGB")
            # Preprocess
            image = self.preprocessor.resize_image(image)

            # Segmentation
            results = self.segmentation.segment_image(image)

            labels = [r["label"] for r in results]

            building_ratio = self.segmentation.get_object_ratio(
                image, results, ["building", "house", "skyscraper"])

            self.logger.info(
                f"Building ratio for {path}: {building_ratio:.4f}")

            image_result["building_ratio"] = float(building_ratio)

            self.logger.info(
                f"Segmentation labels for {path}: {labels}")

            if "building" not in labels and "house" not in labels and "skyscraper" not in labels:
                self.logger.warning(
                    f"No building detected in {path}, marking empty lot."
                )
                image_result["prediction"] = BuildingType.EMPTY_LOT.value
                image_result["building_ratio"] = 0.0
                image_result["confidence"] = 1.0
            elif building_ratio < 0.10:
                self.logger.warning(
                    f"Not enough details to determine building in {path}."
                )
                image_result["prediction"] = BuildingType.UNKNOWN.value
                image_result["building_ratio"] = 0.0
                image_result["confidence"] = 1.0
            else:
                building_mask = self.segmentation.create_object_mask(
                    results,
                    labels=["building", "house", "skyscraper"]
                )
                building_cropped_image = self.segmentation.crop_image_with_mask(
                    image,
                    building_mask
                )

                label, confidence = self.classifier.classify(
                    building_cropped_image)
                image_result["prediction"] = label.value
                image_result["confidence"] = float(confidence)
                image_result["building_ratio"] = float(building_ratio)

                self.logger.info(
                    f"Vision model prediction for {path}: {label} (confidence: {confidence:.4f})")

            return image_result

        except Exception as e:

            self.logger.warning(
                f"Failed processing {path}: {e}"
            )

    def aggregate_predictions(self, year_wise_prediction):

        scores = {}

        years = list(year_wise_prediction.keys())

        min_year = min(years)

        for year, pred_data in year_wise_prediction.items():

            label = pred_data["prediction"]
            confidence = pred_data["confidence"]

            # recency weight
            recency_weight = (year - min_year + 1)

            weight = recency_weight * confidence

            scores[label] = scores.get(label, 0) + weight

        # remove unknown if better options exist
        if "unknown" in scores and len(scores) > 1:
            del scores["unknown"]

        final_prediction = max(scores, key=scores.get)

        return final_prediction, scores
