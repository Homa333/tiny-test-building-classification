import csv
import json


class OutputWriter:

    def __init__(self, logger=None):
        self.logger = logger

    def write_predictions_csv(self, file_path: str, final_results: dict):
        """
        Writes final location-level predictions to CSV
        """

        try:
            with open(file_path, "w", newline="") as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=["location_id", "address", "prediction"]
                )

                writer.writeheader()
                print(final_results)

                for location_id, data in final_results.items():

                    writer.writerow({
                        "location_id": location_id,
                        "address": data.get("address", "unknown"),
                        "prediction": data.get("final_prediction", "unknown")
                    })

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed writing CSV: {e}")
            else:
                print(f"ERROR writing CSV: {e}")

    def write_intermediate_jsonl(self, file_path: str, image_result: dict):
        """
        Writes one intermediate result as a JSON line
        """

        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(image_result) + "\n")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed writing intermediate result: {e}")
            else:
                print(f"WARNING writing JSONL: {e}")
