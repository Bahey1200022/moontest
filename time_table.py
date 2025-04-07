from datetime import datetime

def calc_time(inzone_namess, calc_collections, known_namess):
    current_date = datetime.now().date().isoformat()   # e.g., '2025-04-07'
    current_time = datetime.now().time().isoformat()   # e.g., '15:45:20.123456'
    
    for name in known_namess:
        # Was the person seen in this frame?
        detected = name in inzone_namess
        detected_frames = 1 if detected else 0

        # Check if a document for this person and today's date already exists
        existing_doc = calc_collections.find_one({"name": name, "date": current_date})

        if existing_doc:
            # Person already has a record for today, update it
            update_fields = {
                "$inc": {
                    "total_frames": 1,
                    "detected_frames": detected_frames
                }
            }

            if detected:
                update_fields["$set"] = {"current": current_time}

            calc_collections.update_one(
                {"name": name, "date": current_date},
                update_fields
            )

        else:
            # First time this person is seen today, insert new document
            calc_collections.insert_one({
                "name": name,
                "date": current_date,
                "first_seen": current_time,
                "current": current_time if detected else None,
                "total_frames": 1,
                "detected_frames": detected_frames
            })
