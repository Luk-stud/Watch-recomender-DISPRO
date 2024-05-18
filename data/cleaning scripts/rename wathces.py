import os
import json
import shutil

def make_compatible_with_windows(filename):
    # Replace colons with underscores
    filename = filename.replace(':', '_')
    # Ensure the filename is compatible with Windows
    filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '.', '-', ' ', 'ö'])
    return filename

def rename_files_and_update_json(json_file, new_json_file):
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create a backup of the original JSON data
    original_data = data.copy()
    
    # Track success for each file
    success = True
    
    # List to store paths of successfully renamed files
    renamed_files = []
    
    # Iterate through each item in the JSON
    for item in data:
        image_path = item.get('Image Path')
        if image_path:
            # Check if the file exists
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                success = False
                break
            
            # Extract directory and filename from the image path
            directory, filename = os.path.split(image_path)
            
            # Replace spaces with underscores and update filename
            new_filename = filename.replace(' ', '_')
            
            # Make the filename compatible with Windows
            new_filename = make_compatible_with_windows(new_filename)
            
            # Construct the new image path
            new_image_path = os.path.join(directory, new_filename)
            
            # Attempt to rename the file
            try:
                os.rename(image_path, new_image_path)
                renamed_files.append((image_path, new_image_path))
            except Exception as e:
                print(f"Error renaming file: {e}")
                success = False
                break
    
    if success:
        # Update the JSON with the new image paths
        for item, (old_path, new_path) in zip(data, renamed_files):
            item['Image Path'] = new_path
        
        # Save the modified JSON under the new name
        with open(new_json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    else:
        # Revert changes by restoring the original JSON data and renaming files back
        data = original_data
        for old_path, new_path in renamed_files:
            shutil.move(new_path, old_path)
        
        print("Some files were not renamed. JSON file not updated.")

# Replace 'your_json_file.json' with the path to your JSON file
json_file = 'scraping_output/cleaned_watches.json'
# Replace 'new_json_file.json' with the desired new name for the JSON file
new_json_file = 'scraping_output/renamed watches.json'
rename_files_and_update_json(json_file, new_json_file)
