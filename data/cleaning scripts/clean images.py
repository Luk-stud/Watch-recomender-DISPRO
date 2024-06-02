import os
import json
import pandas as pd

def load_json_data(file_path):
    """ Load JSON data from a file. """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json_data(data, file_path):
    """ Save JSON data to a file. """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def validate_images(data, base_path):
    """Validate image files exist and remove brands with fewer than two valid images."""
    image_count = {}
    valid_data = []

    for item in data:
        image_path = item.get("Image Path", "")
        full_image_path = os.path.join(base_path, image_path)
        brand = item.get("Brand:", "").strip()

        if os.path.exists(full_image_path):
            image_count[brand] = image_count.get(brand, 0) + 1
            valid_data.append(item)
        else:
            print(f"Warning: File not found {full_image_path}")

    return [item for item in valid_data if image_count.get(item.get("Brand:", "").strip(), 0) >= 2]

def clean_keys_and_convert_data(data):
    """ Clean keys by removing colons and convert specified measurement fields. """
    # Create a new list to store cleaned data
    cleaned_data = []
    for item in data:
        # Clean keys by removing colons
        new_item = {key.replace(':', ''): value for key, value in item.items()}
        
        # Replace " mm" and convert Diameter, Height, and Lug Width to float
        for field in ['Diameter', 'Height', 'Lug Width']:
            if field in new_item:
                new_item[field] = float(new_item[field].replace(' mm', ''))
        
        cleaned_data.append(new_item)
    return cleaned_data

if __name__ == "__main__":
    input_json_path = 'data/watches_database_main.json'
    input_json_path2 = 'data/watches_database_main.json'
    output_json_path = 'data/cleaned_watches v3.json'
    image_base_path = 'scraping_output/images'

    # Load the data
    watches_data = load_json_data(input_json_path2)

    # Validate and clean the data
    #cleaned_data = validate_images(watches_data, image_base_path)
    df = pd.DataFrame(clean_keys_and_convert_data(watches_data))
   
    # Save the cleaned data
    cleaned_df_dict = df.to_dict(orient='records')  # Convert DataFrame back to list of dicts for JSON serialization
    save_json_data(cleaned_df_dict, output_json_path)

    print(f"Data cleaned and saved to {output_json_path}. Total entries retained: {len(cleaned_df_dict)}.")