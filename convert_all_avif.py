from PIL import Image
import pillow_avif
import os

base_folder = "dataset"

for category in ["real", "ai"]:
    folder_path = os.path.join(base_folder, category)

    for file in os.listdir(folder_path):
        if file.lower().endswith(".avif"):
            avif_path = os.path.join(folder_path, file)

            try:
                img = Image.open(avif_path)
                new_name = file.replace(".avif", ".jpg")
                jpg_path = os.path.join(folder_path, new_name)

                img.save(jpg_path, "JPEG", quality=95)
                print(f"Converted: {file} → {new_name}")

            except Exception as e:
                print(f"Failed to convert {file}: {e}")

print("Conversion completed.")