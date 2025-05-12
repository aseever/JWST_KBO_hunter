Try to find unidentified Kuiper Belt Objects (KBO) in James Webb Space Telescope "fits" files available for download from MAST

It goes like this:

**# Search MAST for observations in a given box** 

python kbo_hunt.py search --config config/coordinates.txt --ecliptic-priority

**# Filter results (using the actual filenames from the previous step)**

python kbo_hunt.py filter --catalog data/search_20250511_123456/combined_results_20250511_123456.json

**# Download filtered candidates**

python kbo_hunt.py download --catalog data/kbo_candidates_20250511_123456.json

**#post-download filter and line up the files**

python preprocess.py --fits-dir ./data/fits --verbose

**#Look for motion, shift n stack, squint and scrunch up your nose**

python kbo_detector.py --preprocessed-dir ./data/preprocessed --verbose

**#Check for catalog entries of anomalies (apis don't work yet)**

python lookup_kbo_candidates.py --verbose

