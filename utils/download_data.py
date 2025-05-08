import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re


def extract_url_components(base_url):
    parts = base_url.rstrip("/").split("/")
    
    if len(parts) < 6:
        raise ValueError("URL structure is incorrect.")
    
    satellite = parts[-4]
    data_level = parts[-3]
    instrument = parts[-2]
    
    return satellite, data_level, instrument


def download_themis_data(base_url, start_date, end_date, download_dir):
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    satellite, data_level, instrument = extract_url_components(base_url)
    
    if instrument == "state":
        pattern = re.compile(rf"{satellite}_{data_level}_{instrument}_(\d{{8}})_v02\.cdf$")
    else:
        pattern = re.compile(rf"{satellite}_{data_level}_{instrument}_(\d{{8}})(?:_v\d{{2}})?\.cdf$")

    for link in soup.find_all("a"):
        href = link.get("href")
        match = pattern.search(href) if href else None
        if match:
            file_date = match.group(1)
            if start_date <= file_date <= end_date:
                file_url = urljoin(base_url, href)                
                file_path = os.path.join(download_dir, href)

                print(f"Downloading: {file_url}")
                file_response = requests.get(file_url)
                with open(file_path, "wb") as file:
                    file.write(file_response.content)

    print("Download complete.")