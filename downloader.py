import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- CONFIGURATION ---
main_url = "https://miplib.zib.de/tag_binary.html"
COMMON_STRING_PART = "/instance/"  # Put your common string pattern here!

# Create a folder to save files
os.makedirs("downloaded_files", exist_ok=True)

print(f"Fetching main page: {main_url}")
try:
    response = requests.get(main_url, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
except Exception as e:
    print(f"CRITICAL ERROR: Failed to fetch main page: {e}")
    exit()

# Step 2: Find all links on the main page
for tag in soup.find_all('a', href=True):
    href_value = tag['href']
    
    # Check if the common string part exists inside the link
    if COMMON_STRING_PART in href_value:
        page_url = urljoin(main_url, href_value)
        print(f"Matched subpage link: {page_url}")
        
        # Step 3: Visit each matching subpage
        try:
            sub_page = requests.get(page_url, timeout=15)  # Corrected variable name alignment
            sub_page.raise_for_status()
            sub_soup = BeautifulSoup(sub_page.text, 'html.parser')
            
            # Step 4: Find the actual download link inside the sub-page
            for download_tag in sub_soup.find_all('a', href=True):
                file_url = download_tag['href']
                
                if any(file_url.lower().endswith(ext) for ext in ['.mps', '.mps.gz']):
                    full_file_url = urljoin(page_url, file_url)
                    
                    # Step 5: Download the file safely
                    # Clean filename in case URL has trailing parameters (?download=1)
                    raw_filename = os.path.basename(full_file_url)
                    file_name = raw_filename.split('?')[0]
                    
                    if not file_name:
                        file_name = "downloaded_file"
                    
                    print(f"Downloading data from: {full_file_url}")
                    file_data = requests.get(full_file_url, timeout=30).content
                    
                    save_path = os.path.join("downloaded_files", file_name)
                    with open(save_path, "wb") as f:
                        f.write(file_data)
                        
                    print(f"SUCCESS: Downloaded {file_name}")
                    
                    # Once we find the download link on this subpage, 
                    # stop looking at other links on this subpage and move to the next file!
                    break 
                    
        except Exception as e:
            print(f"ERROR: Skipped {page_url} due to error: {e}")

print("All processing finished!")