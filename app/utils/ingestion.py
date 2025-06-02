from bs4 import BeautifulSoup
import requests
from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import logging
import re

MIN_LONGITUD_FRAGMENTO = 30
MAX_LONGITUD_FRAGMENTO = 1000  # caracteres, configurable

def extraer_texto_web(url: str, selectores: List[str] = None) -> List[str]:
    if selectores is None:
        selectores = ['p', 'h1', 'h2', 'h3', 'article', 'section', 'div']

    html = obtener_html_con_requests(url)
    if html is None:
        html = obtener_html_con_requests(url)

    if html is None:
        logging.warning(f"❌ No se pudo obtener contenido de: {url}")
        return []

    fragmentos = extraer_fragmentos(html, selectores)
    fragmentos_limpios = limpiar_fragmentos(fragmentos)

    logging.info(f"📄 {len(fragmentos_limpios)} fragmentos extraídos de: {url}")
    return fragmentos_limpios


def obtener_html_con_requests(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logging.warning(f"⚠️ Error con requests en {url}: {e}")
        return None