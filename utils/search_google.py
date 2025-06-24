# utils/search_google.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def search_google(query: str, num_results: int = 3) -> str:
    headers = {'User-Agent': 'Mozilla/5.0'}
    search_url = f"https://www.google.com/search?q={quote(query)}"
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for g in soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')[:num_results]:
        results.append(g.get_text())

    return "\n".join(results)
