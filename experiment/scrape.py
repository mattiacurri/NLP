import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

def scrape():
    print(" ==== Scraping Empulia Normativa ==== ")
    url = "http://www.empulia.it/tno-a/empulia/Empulia/SitePages/Normativa.aspx"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    links = []
    div = soup.find("div", class_="ms-rte-layoutszone-inner")
    if div:
        for a in tqdm(div.find_all("a", href=True), desc="Processing links", unit="link", position=0, leave=True):
            links.append(a["href"])
            response = requests.get("http://www.empulia.it" + a["href"])
            if response.status_code == 200:
                filename = a["href"].split("/")[-1]
                if os.path.exists(f"docs/{filename}"):
                    print(f" ==== Skipping {filename}, already exists. ====")
                    continue
                with open(f"docs/{filename}", "wb") as f:
                    f.write(response.content)
            else:
                print(f"ERROR: Failed to download {a['href']}")