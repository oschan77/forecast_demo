import os
import random
import shutil


def copy_random_files(source_dir, target_dir, num_files_to_copy):
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory '{source_dir}' does not exist.")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = [
        f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))
    ]

    if num_files_to_copy > len(files):
        raise ValueError(
            f"Number of files to copy ({num_files_to_copy}) is greater than the number of available files ({len(files)})."
        )

    files_to_copy = random.sample(files, num_files_to_copy)

    for file_name in files_to_copy:
        full_file_name = os.path.join(source_dir, file_name)
        shutil.copy(full_file_name, target_dir)
        print(f"Copied {file_name} to {target_dir}")


if __name__ == "__main__":
    source_dir = "data/processed/walmart_sales/BySD"
    target_dir = "data/processed/walmart_sales/BySD500"
    num_files_to_copy = 500

    copy_random_files(source_dir, target_dir, num_files_to_copy)
