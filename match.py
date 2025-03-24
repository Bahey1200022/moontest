from scipy.spatial import distance

def map_safety_to_overall(avg_safety, overall_avg):
    safety_items = [item for item in avg_safety if item['class'] not in ['NO-Mask', 'Person']]
    
    safety_mappings = {
        unknown: {
            'bounding_box_avg': overall_avg[unknown],
            'safety_info': {'Hardhat': None, 'NO-Hardhat': None, 'Safety Vest': None, 'NO-Vest': None}
        }
        for unknown in overall_avg
    }
    
    for safety_item in safety_items:
        closest_unknown = min(overall_avg, key=lambda u: abs(overall_avg[u] - safety_item['bounding_box_avg']))
        safety_mappings[closest_unknown]['safety_info'][safety_item['class']] = safety_item['bounding_box_avg']
    
    return safety_mappings

def display_mappings(mappings):
    for unknown, data in mappings.items():
        print(f"- **{unknown}**")
        print(f"  - Bounding Box Avg: **{data['bounding_box_avg']}**")
        for key, value in data['safety_info'].items():
            print(f"  - {key}: **{value}**")
        print("")
