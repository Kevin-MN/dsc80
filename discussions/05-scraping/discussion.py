# discussion.py


import os
import numpy as np
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_website_urls():
    """
    Get all the website URLs

    :Example:
    >>> urls = get_website_urls()
    >>> len(urls)
    50
    >>> 'catalogue' in urls[0]
    True
    """
    urls = []
    for i in np.arange(1,51):
        urls.append('http://books.toscrape.com/catalogue/page-' + str(i) + '.html')
    
    return urls


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def book_categories():
    """
    Get all the book categories and return them as a list

    :Example:
    >>> categories = book_categories()
    >>> len(categories)
    50
    >>> 'Classics' in categories
    True
    """
    home_page_soup = BeautifulSoup(requests.get('http://books.toscrape.com/index.html').text)
    cats_raw = home_page_soup.find('div', {'class':'side_categories'}).find_all('a')

    categories = []
    for i in cats_raw:
        categories.append(i.text.replace('\n','').strip(' '))
    
    return categories[1:len(categories)]
    
