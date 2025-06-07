from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, SoupStrainer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def obtener_urls_del_html(html, base_url):
    soup = BeautifulSoup(html, "html.parser", parse_only=SoupStrainer("a"))
    urls = set()
    for tag in soup:
        if tag.name == "a" and tag.get("href"):
            href = urljoin(base_url, tag["href"])
            if urlparse(href).netloc == urlparse(base_url).netloc:
                urls.add(href.split("#")[0])
    return urls

def analizar_enlaces(url_base):
    print(f"🌐 Visitando {url_base}")
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url_base)
        driver.implicitly_wait(8)
        html = driver.page_source
        urls = obtener_urls_del_html(html, url_base)
        print(f"🔗 Total enlaces internos encontrados: {len(urls)}\n")
        for u in sorted(list(urls))[:30]:
            print(" -", u)
    finally:
        driver.quit()

if __name__ == "__main__":
    analizar_enlaces("https://www.onda.es")
