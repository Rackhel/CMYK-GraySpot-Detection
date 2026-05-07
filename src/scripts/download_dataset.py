import os
import gdown
import zipfile

DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "images.zip")

# Use the direct download URL format
FILE_ID = "1URGC25XOhgaZysv5PqViKaOOJwi2S1iT"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

os.makedirs(DATA_DIR, exist_ok=True)

# Check if already extracted
if not os.path.exists(os.path.join(DATA_DIR, "images")):
    print("Downloading dataset...")
    
    # Download with fuzzy=True to handle Google Drive links better
    gdown.download(URL, ZIP_PATH, quiet=False, fuzzy=True)
    
    # Check if download was successful (file should be > 1MB)
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 1000000:
        print("Extracting dataset...")
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            print("Dataset ready.")
            
            # Optionally remove the zip file after extraction
            # os.remove(ZIP_PATH)
            # print("Cleaned up zip file")
        except zipfile.BadZipFile:
            print("Error: Downloaded file is not a valid ZIP file")
            print(f"File size: {os.path.getsize(ZIP_PATH)} bytes")
            print("The file might be too small - possibly an HTML error page")
    else:
        print(f"Error: Downloaded file is too small or doesn't exist")
        print(f"File size: {os.path.getsize(ZIP_PATH) if os.path.exists(ZIP_PATH) else 'N/A'} bytes")
else:
    print("Dataset already exists. Skipping download.")