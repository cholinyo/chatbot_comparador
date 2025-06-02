from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import requests
from urllib.parse import urljoin, urlparse

# Configuración del navegador sin interfaz (headless)
def get_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    return driver

# Extrae texto limpio desde una URL

def extraer_texto(url):
    try:
        driver = get_driver()
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        html = driver.page_source
        driver.quit()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n").strip()
    except Exception as e:
        return ""

# Extrae enlaces del mismo dominio

def obtener_enlaces(url):
    try:
        driver = get_driver()
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
        html = driver.page_source
        driver.quit()
        soup = BeautifulSoup(html, "html.parser")
        enlaces = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            absoluto = urljoin(url, href)
            if urlparse(absoluto).netloc == urlparse(url).netloc:
                enlaces.append(absoluto)
        return list(set(enlaces))
    except Exception:
        return []
