import zipfile
import os
import sys
import shutil

def unzip_pallets():
    if len(sys.argv) < 2:
        print("Usage: python unzip_pallets.py <path_to_zip_file>")
        sys.exit(1)

    zip_path = sys.argv[1]
    output_dir = "data"
    target_dir_name = "Pallets"

    os.makedirs(output_dir, exist_ok=True)

    temp_extract_path = os.path.join(output_dir, "temp_extract")

    print(f"Unzipping {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_path)

    extracted_items = os.listdir(temp_extract_path)
    if len(extracted_items) != 1:
        print("Warning: Zip archive contains multiple top-level items.")
    extracted_root = os.path.join(temp_extract_path, extracted_items[0])

    final_path = os.path.join(output_dir, target_dir_name)

    if os.path.exists(final_path):
        shutil.rmtree(final_path)

    shutil.move(extracted_root, final_path)
    shutil.rmtree(temp_extract_path)

    print(f"Unzipped and moved to {final_path}")

if __name__ == "__main__":
    unzip_pallets()
