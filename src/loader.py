import pandas as pd
from pathlib import Path


class DataLoader:

    def __init__(self, data_dir: str, metadata_file_name: str = "metadata.csv"):

        self.image_root = Path(data_dir)
        metadata_path = self.image_root / metadata_file_name

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}")

        try:
            self.df = pd.read_csv(metadata_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata CSV: {e}")

    def load_locations(self):

        locations = []

        for _, row in self.df.iterrows():

            folder = self.image_root / row["folder_name"]

            if not folder.exists():
                print(f"WARNING: Missing folder {folder}")
                continue

            images = []

            try:
                files = list(folder.iterdir())
            except Exception as e:
                print(f"WARNING: Cannot access folder {folder}: {e}")
                continue

            for file in files:

                if file.suffix.lower() not in [".jpg", ".png"]:
                    continue

                filename = file.name

                try:
                    year = int(filename.split("-")[0])
                except Exception:
                    print(
                        f"WARNING: Cannot parse year from filename {filename}")
                    continue

                images.append({
                    "year": year,
                    "path": file
                })

            if not images:
                print(f"WARNING: No valid images found in {folder}")
                continue

            # sort by year
            images.sort(key=lambda x: x["year"])

            locations.append({
                "location_id": row.get("location_id", "unknown"),
                "address": row.get("address", ""),
                "folder_name": row.get("folder_name", ""),
                "images": images
            })

        return locations
