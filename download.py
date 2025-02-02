from google_image_fetcher.google_image_fetcher import GoogleImageFetcher
from PIL import Image
import os

allowed_extensions = {".png", ".jpeg", ".jpg", ".webp"}


query = "glasses"
folder_path = "glasses_filtered/validation/glasses/new"



fetcher = GoogleImageFetcher()
fetcher.fetch_images(query, save_folder=folder_path, limit=200, skip=220)

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Get the file extension
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Check if the file extension is allowed
    if file_extension not in allowed_extensions:
        print(f"Invalid file extension: {file_path}")
        os.remove(file_path)
        print(f"Deleted: {file_path}")
        continue  # Skip to the next file
    
    try:
        # Attempt to open the image using PIL
        with Image.open(file_path) as img:
            # Verify that the image is valid
            img.verify()  # Verify that the file is a valid image
    except Exception as e:
        # If an exception occurs, the image is broken
        print(f"Broken image: {file_path} - {e}")
        
        # Delete the broken image
        os.remove(file_path)
        print(f"Deleted: {file_path}")