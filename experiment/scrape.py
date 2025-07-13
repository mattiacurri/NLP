import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

def scrape():
    print(" ==== Scraping Empulia Normativa ==== ")
    url = "http://www.empulia.it/tno-a/empulia/Empulia/SitePages/Normativa.aspx"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Step 1. Find all PDF links
    links = soup.find_all('a', href=True)
    pdf_links = [link['href'] for link in links if link['href'].endswith('.pdf')]
    print(f"Found {len(pdf_links)} PDF links.")

    # Step 2. Download each PDF
    os.makedirs("docs", exist_ok=True)
    
    for pdf_link in tqdm(pdf_links, desc="Downloading PDFs", unit="file", position=0, leave=True):
        pdf_url = "http://www.empulia.it" + pdf_link

        if os.path.exists(os.path.join("docs", os.path.basename(pdf_url))):
            print(f" ==== Skipping {pdf_url}, already downloaded. ====")
            continue
        
        try:
            pdf_response = requests.get(pdf_url)
            with open(os.path.join("docs", os.path.basename(pdf_url)), 'wb') as f:
                f.write(pdf_response.content)
        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")
    
scrape()