from datetime import datetime
from scipy.spatial import distance

def map_safety_to_overall(avg_safety, overall_avg):
    safety_items = [item for item in avg_safety if item['class'] not in ['NO-Mask', 'Person']]
    
    safety_mappings = {
        unknown: {
            'bounding_box_avg': overall_avg[unknown],
            'safety_info': {'Hardhat': None, 'NO-Hardhat': None, 'Safety Vest': None, 'NO-Vest': None},
            'uniform': "No"  # Default to No
        }
        for unknown in overall_avg
    }

    for safety_item in safety_items:
        closest_unknown = min(overall_avg, key=lambda u: abs(overall_avg[u] - safety_item['bounding_box_avg']))
        safety_mappings[closest_unknown]['safety_info'][safety_item['class']] = safety_item['bounding_box_avg']

    # Set uniform = "Yes" if Hardhat or Safety Vest exists
    for unknown, data in safety_mappings.items():
        has_hardhat = data['safety_info']['Hardhat'] is not None
        has_vest = data['safety_info']['Safety Vest'] is not None

        if has_hardhat or has_vest:
            data['uniform'] = "Yes"

    return safety_mappings


def display_mappings(mappings,collections):
    for unknown, data in mappings.items():
        print(f"- {unknown}")
        print(f"  - Bounding Box Avg: **{data['bounding_box_avg']}**")
        for key, value in data['safety_info'].items():
            print(f"  - {key}: {value}")
        print(f"  - Uniform: {data['uniform']}\n")
        date_str = data.get("date", datetime.today().strftime("%Y-%m-%d"))
        try:
            if unknown.lower().startswith("unknown"):
                print(f"Skipping {unknown}")
                continue   
            update_appearance(unknown, date_str, data["uniform"], collections)
        except Exception as e:
            print(f"Error calling update_appearance for {unknown}: {e}")
        



def map_cigs_to_person(cigs, overall_avg):
    # Create a list to store the mappings of each cig to the corresponding person
    mapped_cigs = []

    # Iterate over each cig
    for cig in cigs:
        # Calculate the absolute difference between cig and each person's value in overall_avg
        distances = {person: abs(cig - value) for person, value in overall_avg.items()}

        # Find the person with the smallest distance
        closest_person = min(distances, key=distances.get)
        mapped_cigs.append((cig, closest_person))

    return mapped_cigs




def update_appearance(name, date_str, uniform,collections):
    # Convert date to a consistent format (optional, ISO string)
    date = datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
    update_fields = {
        "$inc": {"total_appearances": 1}
    }

    if uniform.lower() == "yes":
        update_fields["$inc"]["detected_appearances"] = 1

    collections.update_one(
        {"name": name, "date": date},
        update_fields,
        upsert=True
    )
    
    
    