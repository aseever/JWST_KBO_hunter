import csv
import json
import sys

def convert_mast_csv_to_kbo_json(csv_file, json_file):
    """Convert MAST CSV to JSON format expected by kbo_hunt.py"""
    # Field name mapping (MAST field → kbo_hunt field)
    field_map = {
        'ra': 'ra',               # Replace with actual MAST field name for RA
        'dec': 'dec',             # Replace with actual MAST field name for Dec
        'obs_start_date': 't_min',# Replace with actual MAST field name for observation time
        'instrument': 'instrument_name',
        # Add more mappings as needed
    }
    
    observations = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        # Skip initial comment lines starting with #
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        
        # Go back to beginning of file
        f.seek(0)
        
        # Skip comment lines again to get to data
        reader = csv.DictReader((line for line in f if not line.startswith('#')))
        
        for row in reader:
            # Create observation with mapped field names
            obs = {}
            for mast_field, kbo_field in field_map.items():
                if mast_field in row:
                    obs[kbo_field] = row[mast_field]
            
            # Additional fields can be copied as-is
            for key, value in row.items():
                if key not in field_map and value:
                    obs[key] = value
            
            observations.append(obs)
    
    # Format expected by kbo_hunt.py
    catalog = {
        "timestamp": "2025-05-13T00:00:00",
        "observations": observations
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"Converted {len(observations)} observations to {json_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_mast_csv.py input.csv output.json")
        sys.exit(1)
    
    convert_mast_csv_to_kbo_json(sys.argv[1], sys.argv[2])