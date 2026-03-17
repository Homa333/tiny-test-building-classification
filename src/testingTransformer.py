from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from transformers import pipeline
from PIL import Image
import numpy as np

"""
This is a testing file to try out the different models for object detection and segmentation.
"""

# Load the zero-shot object detection pipeline
detector = pipeline(task="zero-shot-object-detection",
                    model="google/owlvit-base-patch32")

segmenter = pipeline("image-segmentation",
                     model="nvidia/segformer-b2-finetuned-ade-512-512")


# Load your street view image
image = Image.open(
    "tiny_gsv_dataset/8140_MT_HOLLY_RD_CHARLOTTE_NC/2022-11.jpg").convert("RGB")

print("Running zero-shot object detection...", image)
# Just tell it what you want to find!
labels_to_find = ["front door", "residential window",
                  "storefront window", "business sign"]

# results = detector(image, candidate_labels=labels_to_find)

results = segmenter(image)

print("Detection results:", results)

for result in results:
    # This will output things like 'building', 'road', 'windowpane'
    print(f"Detected area: {result['label']}")


print(image.size)

image_np = np.array(image)
segmented = image_np.copy()

legend = {}

legend_patches = []

for r in results:
    mask = np.array(r["mask"])
    label = r["label"]

    color = np.random.randint(0, 255, 3)

    segmented[mask > 0] = (
        0.5 * segmented[mask > 0] + 0.5 * color
    )

    legend[label] = color / 255

for label, color in legend.items():
    patch = mpatches.Patch(color=color, label=label)
    legend_patches.append(patch)

plt.figure(figsize=(8, 8))
plt.imshow(segmented)
plt.axis("off")
plt.title("Segmentation Overlay")
plt.legend(
    handles=legend_patches,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.
)

# # legend
# handles = [
#     plt.Rectangle((0, 0), 1, 1, color=legend[label])
#     for label in legend
# ]

# plt.legend(handles, legend.keys(),
#            bbox_to_anchor=(1.05, 1),
#            loc="upper left")

# plt.show()

# for prediction in results:
#     print(
#         f"Found {prediction['label']} with confidence {prediction['score']:.3f} at {prediction['box']}")


building_mask = None

signboard_mask = None

for r in results:
    if r["label"] in ["building", "house", "skyscraper"]:
        mask = np.array(r["mask"])

        if building_mask is None:
            building_mask = mask
        else:
            building_mask = np.logical_or(building_mask, mask)

for r in results:
    if r["label"] in ["signboard", "trade name"]:
        mask = np.array(r["mask"])

        if signboard_mask is None:
            signboard_mask = mask
        else:
            signboard_mask = np.logical_or(signboard_mask, mask)

plt.figure(figsize=(6, 6))
plt.imshow(image_np)
plt.axis("off")
plt.title("main image")
plt.show()

if building_mask is not None:

    coords = np.column_stack(np.where(building_mask))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Optional padding so the crop isn't too tight
    pad = 20
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(image_np.shape[0], y_max + pad)
    x_max = min(image_np.shape[1], x_max + pad)

    cropped_building = image_np[y_min:y_max, x_min:x_max]

    # visualize cropped building
    plt.figure(figsize=(6, 6))
    plt.imshow(cropped_building)
    plt.axis("off")
    plt.title("Cropped  Building")
    plt.show()

else:
    print("No building detected.")

if signboard_mask is not None:

    coords = np.column_stack(np.where(signboard_mask))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Optional padding so the crop isn't too tight
    pad = 20
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(image_np.shape[0], y_max + pad)
    x_max = min(image_np.shape[1], x_max + pad)

    cropped_signboard = image_np[y_min:y_max, x_min:x_max]

    # visualize cropped signboard
    plt.figure(figsize=(6, 6))
    plt.imshow(cropped_signboard)
    plt.axis("off")
    plt.title("Cropped Signboard")
    plt.show()

else:
    print("No building detected.")

# ocr = OCRExtractor()


# gray = cv2.cvtColor(cropped_signboard, cv2.COLOR_RGB2GRAY)
# equalized = cv2.equalizeHist(gray)

# edges = cv2.Canny(gray, 60, 180)

# _, thresh = cv2.threshold(gray, 60, 180, cv2.THRESH_BINARY)

# _, thresh_inv = cv2.threshold(gray, 50, 150, cv2.THRESH_BINARY_INV)

image_np = np.array(image)
# cv2.imshow('Image Window', image_np)
# cv2.imshow('Gray', gray)
# cv2.imshow('equal', equalized)
# cv2.imshow('edges', edges)
# cv2.imshow('thresh', thresh)
# cv2.imshow('thresh_invs', thresh_inv)

# # # Wait indefinitely for a key press (0 means wait forever)
# # # This keeps the window open until the user presses a key
# cv2.waitKey(0)

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# labels = [
#     "a building with balconies",
#     "a building without balconies",
# ]

# # run
# inputs = processor(text=labels, images=cropped_building,
#                    return_tensors="pt", padding=True)
# with torch.no_grad():
#     probs = model(**inputs).logits_per_image.softmax(dim=1).squeeze()

# # print results
# for label, prob in zip(labels, probs):
#     print(f"{label}: {prob:.4f}")

# print("\nResult:", labels[probs.argmax()])

# text = ocr.extract(equalized)
# print(text)
