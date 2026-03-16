import pandas as pd
from pathlib import Path


class DataLoader:

    def __init__(self, data_dir: str, metadata_file_name: str = "metadata.csv"):
        self.df = pd.read_csv(data_dir + metadata_file_name)
        self.image_root = Path(data_dir)

    def load_locations(self):

        locations = []

        for _, row in self.df.iterrows():

            folder = self.image_root / row["folder_name"]

            images = []

            for file in folder.iterdir():

                if file.suffix.lower() not in [".jpg", ".png"]:
                    continue

                filename = file.name

                year = int(filename.split("-")[0])

                images.append({
                    "year": year,
                    "path": file
                })

            # sort by year
            images.sort(key=lambda x: x["year"])

            locations.append({
                "location_id": row["location_id"],
                "address": row["address"],
                "folder_name": row["folder_name"],
                "images": images
            })
        return locations
