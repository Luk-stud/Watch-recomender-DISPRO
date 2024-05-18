import json

def normalize_text(text):
    """ Normalize text to lower case for case-insensitive comparison. """
    return text.lower().strip()

# Load the watch collections JSON
collections = [
    # Your collections JSON here
    {
        "Collection": "Luxury Classics",
        "Watches": [
            {"Brand Name": "Patek Philippe", "Model": "Calatrava", "Reference": "5196J"},
            {"Brand Name": "Vacheron Constantin", "Model": "Patrimony", "Reference": "81180/000P-9539"},
            {"Brand Name": "Audemars Piguet", "Model": "Royal Oak", "Reference": "15400ST.OO.1220ST.01"},
            {"Brand Name": "Piaget", "Model": "Altiplano", "Reference": "G0A29113"}
        ]
    },
    {
        "Collection": "Sports Enthusiasts",
        "Watches": [
            {"Brand Name": "Omega", "Model": "Aqua Terra", "Reference": "220.12.41.21.06.001"},
            {"Brand Name": "TAG Heuer", "Model": "Aquaracer", "Reference": "WAY201A.BA0927"},
            {"Brand Name": "Rolex", "Model": "Submariner", "Reference": "126610LV"},
            {"Brand Name": "Panerai", "Model": "Luminor", "Reference": "PAM01028"}
        ]
    },
    {
        "Collection": "Adventure and Outdoor",
        "Watches": [
            {"Brand Name": "Hamilton", "Model": "Khaki Field", "Reference": "H69439931"},
            {"Brand Name": "Garmin", "Model": "Fenix", "Reference": "Fenix 6X Pro"},
            {"Brand Name": "Suunto", "Model": "Core", "Reference": "SS014279010"},
            {"Brand Name": "Casio", "Model": "G-Shock", "Reference": "GW-9400-1"}
        ]
    },
    {
        "Collection": "Modern Innovators",
        "Watches": [
            {"Brand Name": "Hublot", "Model": "Big Bang", "Reference": "411.JX.4802.RT"},
            {"Brand Name": "Ulysse Nardin", "Model": "Freak", "Reference": "2505-250"},
            {"Brand Name": "Richard Mille", "Model": "RM 005", "Reference": "RM005"},
            {"Brand Name": "Roger Dubuis", "Model": "Excalibur", "Reference": "DBEX0577"}
        ]
    },
    {
        "Collection": "Dress Watches",
        "Watches": [
            {"Brand Name": "Jaeger-LeCoultre", "Model": "Reverso", "Reference": "3958420"},
            {"Brand Name": "Longines", "Model": "Heritage Classic", "Reference": "L2.828.4.73.0"},
            {"Brand Name": "Montblanc", "Model": "Heritage Chronometrie", "Reference": "112533"},
            {"Brand Name": "Girard-Perregaux", "Model": "1966", "Reference": "49525-52-131-BK6A"}
        ]
    }
]

# Load your watch database JSON (assuming it's stored in a file)
with open('scraping_output/wathec.json', 'r') as file:
    watch_database = json.load(file)


def find_watch_details(brand, reference, name):
    """
    Find a watch in the database that matches the brand and partially matches
    the reference number and name. Returns the first match found.
    """
    normalized_ref = normalize_text(reference)
    normalized_name = normalize_text(name)
    for watch in watch_database:
        db_brand = normalize_text(watch.get('Brand:', ''))
        db_reference = normalize_text(watch.get('Reference:', ''))
        db_name = normalize_text(watch.get('Name:', ''))
        if db_brand == brand and normalized_ref in db_reference and normalized_name in db_name:
            return watch
    return None

# Update the collections with additional information from the database
updated_collections = []
for collection in collections:
    new_collection = {
        'Collection': collection['Collection'],
        'Watches': []
    }
    for watch in collection['Watches']:
        details = find_watch_details(normalize_text(watch['Brand Name']), watch['Reference'], watch['Model'])
        if details:
            # Append all details from the database entry to the watch
            watch['Extra Details'] = details
        new_collection['Watches'].append(watch)
    updated_collections.append(new_collection)

# Output the updated collections to a new JSON file
with open('val_data_by_aestetic.json', 'w') as outfile:
    json.dump(updated_collections, outfile, indent=4)

print("Updated collections have been saved to 'val_data_by_aestetic.json'.")