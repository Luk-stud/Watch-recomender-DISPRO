import os
import json
import shutil

def make_compatible_with_windows(filename):
    # Windows filename forbidden characters
    forbidden_chars = ':*?"<>|\\/'
    for char in forbidden_chars:
        filename = filename.replace(char, '_')
    return filename

def sanitize_folder_name(folder_name):
    # Sanitize folder names by replacing problematic characters
    forbidden_chars = ':*?"<>|\\/'
    for char in forbidden_chars:
        folder_name = folder_name.replace(char, '_')
    # Windows does not support folder names ending with a '.' or space
    return folder_name.rstrip('. ').replace(' ', '_')

def rename_files_and_update_json(json_file, new_json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        original_image_path = item['Image Path']
        parts = original_image_path.split('/')
        new_parts = [sanitize_folder_name(part) if i != len(parts) - 1 else make_compatible_with_windows(part) for i, part in enumerate(parts)]
        sanitized_path = '/'.join(new_parts)

        if original_image_path != sanitized_path:
            directory = os.path.dirname(sanitized_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            try:
                shutil.move(original_image_path, sanitized_path)
                item['Image Path'] = sanitized_path
            except Exception as e:
                print(f"Failed to rename {original_image_path} to {sanitized_path}: {str(e)}")
                continue

    with open(new_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def delete_empty_folders(root_dir):
    # Counter to track the number of deleted folders
    deleted_count = 0

    # Walk through all the directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # topdown=False makes os.walk() yield the sub-directories of a directory
        # after it has yielded all of its immediate sub-directories which is necessary for deleting.

        # Check if the directory is empty
        if not dirnames and not filenames:
            # Try to remove empty directories, catch any exceptions like if the directory is not empty
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty folder: {dirpath}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {dirpath}: {e}")

    return deleted_count

# Replace 'path_to_start_directory' with the path of the directory you want to start from
root_directory = 'scraping_output/images'
deleted_folders = delete_empty_folders(root_directory)
print(f"Total empty folders deleted: {deleted_folders}")

# Set the path to your JSON files
json_file = 'scraping_output/renamed watches.json'
new_json_file = 'scraping_output/updated_completed_log.json'
rename_files_and_update_json(json_file, new_json_file)
