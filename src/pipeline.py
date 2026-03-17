import os
from PIL import Image, UnidentifiedImageError

from src.output_writer import OutputWriter
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
        self.output_writer = OutputWriter(logger)

        self.prediction_file = os.path.join(output_dir, "predictions.csv")
        self.intermediate_file = os.path.join(
            output_dir, "intermediate_results.jsonl"
        )

    def run(self):

        self.logger.info("Loading metadata")

        try:
            locations = self.loader.load_locations()
        except Exception as e:
            self.logger.error(f"Failed loading metadata: {e}")
            return

        final_results = {}

        for location in locations:

            location_id = location["location_id"]
            self.logger.info(f"Processing {location_id}")

            images = location.get("images", [])

            final_results[location_id] = {
                "year_wise_prediction": {},
                "address": location.get("address", "unknown")
            }

            for img_info in images:

                year = img_info["year"]
                path = img_info["path"]

                if not os.path.exists(path):
                    self.logger.warning(f"Missing image: {path}")
                    continue

                image_result = {"image_path": str(path)}

                result = self.classify_building(path)

                if result is None:
                    self.logger.warning(f"Skipping failed image: {path}")
                    continue

                image_result.update(result)

                final_results[location_id]["year_wise_prediction"][year] = image_result

            try:

                final_prediction, scores = self.aggregate_predictions(
                    final_results[location_id]["year_wise_prediction"]
                )

                final_results[location_id]["final_prediction"] = final_prediction
                final_results[location_id]["scores"] = scores

            except Exception as e:

                self.logger.warning(
                    f"Aggregation failed for {location_id}: {e}"
                )

                final_results[location_id]["final_prediction"] = BuildingType.UNKNOWN.value
                final_results[location_id]["scores"] = {}

        self.output_writer.write_predictions_csv(
            self.prediction_file, final_results)
        self.output_writer.write_intermediate_jsonl(
            self.intermediate_file, final_results)

        self.logger.info("Pipeline completed")

    def classify_building(self, path: str) -> dict:

        try:

            image_result = {}

            try:
                image = Image.open(path).convert("RGB")
            except FileNotFoundError:
                self.logger.warning(f"Image not found: {path}")
                return None
            except UnidentifiedImageError:
                self.logger.warning(f"Corrupted image file: {path}")
                return None

            # Preprocess
            image = self.preprocessor.resize_image(image)

            # Segmentation
            try:
                results = self.segmentation.segment_image(image)
            except Exception as e:
                self.logger.warning(f"Segmentation failed for {path}: {e}")
                return None

            if not results:
                self.logger.warning(f"No segmentation output for {path}")
                return None

            labels = [r["label"] for r in results]

            building_ratio = self.segmentation.get_object_ratio(
                image, results, ["building", "house", "skyscraper"]
            )

            self.logger.info(
                f"Building ratio for {path}: {building_ratio:.4f}"
            )

            image_result["building_ratio"] = float(building_ratio)

            self.logger.info(
                f"Segmentation labels for {path}: {labels}"
            )

            if not any(label in labels for label in ["building", "house", "skyscraper"]):

                self.logger.warning(
                    f"No building detected in {path}, marking empty lot."
                )

                image_result["prediction"] = BuildingType.EMPTY_LOT.value
                image_result["confidence"] = 1.0
                image_result["building_ratio"] = 0.0

                return image_result

            if building_ratio < 0.10:

                self.logger.warning(
                    f"Insufficient visual information in {path}"
                )

                image_result["prediction"] = BuildingType.UNKNOWN.value
                image_result["confidence"] = 1.0

                return image_result

            try:

                building_mask = self.segmentation.create_object_mask(
                    results,
                    labels=["building", "house", "skyscraper"]
                )

                building_cropped_image = self.segmentation.crop_image_with_mask(
                    image,
                    building_mask
                )

            except Exception as e:

                self.logger.warning(
                    f"Mask creation failed for {path}: {e}"
                )

                return None

            try:

                label, confidence = self.classifier.classify(
                    building_cropped_image
                )

            except Exception as e:

                self.logger.warning(
                    f"Vision model failed for {path}: {e}"
                )

                image_result["prediction"] = BuildingType.UNKNOWN.value
                image_result["confidence"] = 0.0

                return image_result

            image_result["prediction"] = label.value
            image_result["confidence"] = float(confidence)

            self.logger.info(
                f"Vision model prediction for {path}: {label} (confidence: {confidence:.4f})"
            )

            return image_result

        except Exception as e:

            self.logger.exception(
                f"Unexpected failure while processing {path}: {e}"
            )

            return None

    def aggregate_predictions(self, year_wise_prediction):

        if not year_wise_prediction:
            return BuildingType.UNKNOWN.value, {}

        scores = {}

        years = list(year_wise_prediction.keys())
        min_year = min(years)

        for year, pred_data in year_wise_prediction.items():

            label = pred_data.get("prediction", BuildingType.UNKNOWN.value)
            confidence = pred_data.get("confidence", 0.0)

            recency_weight = (year - min_year + 1)

            weight = recency_weight * confidence

            scores[label] = scores.get(label, 0) + weight

        if "unknown" in scores and len(scores) > 1:
            del scores["unknown"]

        final_prediction = max(scores, key=scores.get)

        return final_prediction, scores
