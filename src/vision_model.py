from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from src.building_type_enum import BuildingType


class VisionModel:

    def __init__(self):

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")

        self.prompts = {
            "single_family": [
                "a detached single family house",
                "a suburban house with front yard",
                "a house with pitched roof",
            ],
            "apartment_condo": [
                "a tall apartment building",
                "a multi story residential building",
                "a condominium building with many windows",
            ],
            "commercial": [
                "a commercial building with storefronts",
                "a retail store or shop",
                "an office building",
            ],
            "mixed_use": [
                "a building with shops on ground floor and apartments above",
                "a mixed use building with retail and residential",
            ],
        }

        all_prompts = []
        prompt_to_category = {}
        for category, prompts in self.prompts.items():
            for prompt in prompts:
                all_prompts.append(prompt)
                prompt_to_category[prompt] = category

        self.all_prompts = all_prompts
        self.prompt_to_category = prompt_to_category

    def classify(self, image: Image.Image) -> tuple[BuildingType, float]:
        inputs = self.processor(
            text=self.all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            probs = self.model(
                **inputs).logits_per_image.softmax(dim=1).squeeze()

        scores = dict(zip(self.all_prompts, probs.tolist()))

        # aggregate scores per category
        category_scores = {cat: 0.0 for cat in self.prompts}
        for prompt, score in scores.items():
            category_scores[self.prompt_to_category[prompt]] += score

        best_category = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category]

        return BuildingType(best_category), round(best_score, 4)
