import requests
import os
from tqdm import tqdm  # Import tqdm for progress bar
import zipfile

url = "https://zenodo.org/api/records/2535967/files-archive"

save_path = "../DATA/CIFAR-10-C/cifar.zip"
# Send a GET request to download the file with streaming enabled
response = requests.get(url, stream=True)

# # Send a GET request to download the file with streaming enabled
# response = requests.get(url, stream=True)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the total file size from the response headers
    total_size = int(response.headers.get('content-length', 0))
    
    # Check Content-Type header to confirm the file type is zip
    content_type = response.headers.get('Content-Type', '')
    if 'zip' not in content_type:
        print("Warning: The file is not a zip file. Content-Type:", content_type)
    
    # Create a progress bar using tqdm
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as bar:
        with open(save_path, 'wb') as file:
            # Download the file in chunks
            for chunk in response.iter_content(chunk_size=1024):  # 1 KB per chunk
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))  # Update the progress bar with the chunk size
    print(f"File downloaded successfully and saved as {save_path}")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

# Now check if the file is a valid zip file before extracting
if save_path.endswith('.zip'):
    if zipfile.is_zipfile(save_path):  # Check if it's a valid zip file
        try:
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall('cifar10-c')  # You can specify a different folder if needed
            print("File extracted successfully.")
        except zipfile.BadZipFile:
            print(f"Error: The file '{save_path}' is not a valid zip file or is corrupted.")
    else:
        print(f"Error: '{save_path}' is not recognized as a valid zip file.")
else:
    print(f"Error: The file at '{save_path}' does not have a .zip extension.")