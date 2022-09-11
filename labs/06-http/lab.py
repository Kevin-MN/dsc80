# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    >>> question1()
    >>> os.path.exists('lab06_1.html')
    True
    """
    # Don't change this function body!
    # No python required; create the HTML file.

    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------




def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    raw = bs4.BeautifulSoup(text, features='lxml')
    raws = raw.find_all('article', {'class':'product_pod'})

    cleared = []
    for i in raws:
        if ((i.find('p').get('class')[1] == 'Four') | (i.find('p').get('class')[1] == 'Five')) & (float(i.find('p',{'class':'price_color'}).text.replace('Â','').replace('£','')) < 50):
            cleared.append(i.find('a').get('href'))

    return cleared


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    test2 = bs4.BeautifulSoup(text, features='lxml')

    if test2.find('ul',{'class':'breadcrumb'}).find_all('a')[2].text in categories:
        stats = test2.find_all('tr')
        upc = stats[0].text.replace('\n','').replace('UPC',"")
        prod_type = stats[1].text.replace('\n','').replace('Product Type',"")
        price_no_tax = stats[2].text.replace('\n','').replace('Price (excl. tax)',"")
        price_w_tax = stats[3].text.replace('\n','').replace('Price (incl. tax)',"")
        tax = stats[4].text.replace('\n','').replace('Tax',"")
        avail = stats[5].text.replace('\n','').replace('Availability',"")
        num_rev  = stats[6].text.replace('\n','').replace('Number of reviews',"")


        cat = test2.find('ul',{'class':'breadcrumb'}).find_all('a')[2].text
        #cat

        rate = test2.find('div', {'class':'col-sm-6 product_main'}).find_all('p')[2].get('class')[1]
        #rate

        desc = test2.find_all('p')[3].text
        #desc

        title = test2.find('ul',{'class':'breadcrumb'}).find_all('li')[3].text
        #title

        return dict({'UPC':upc, 'Product Type':prod_type,'Price (excl. tax)':price_no_tax, 'Price (incl. tax)':price_w_tax,'Tax':tax,'Availability':avail,'Number of reviews':num_rev,'Category':cat,'Rating':rate,'Description':desc,'Title':title})
    else:
        return None

def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).
    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """
    
    #k = 50

    #categories = ['Default,Romance','Mystery']
    df = pd.DataFrame(columns = ['UPC','Product Type','Price (excl. tax)', 'Price (incl. tax)','Tax','Availability','Number of reviews','Category','Rating','Description','Title'])

    for i in np.arange(1,k+1):
        text2 = requests.get('https://books.toscrape.com/catalogue/page-' + str(i) + '.html')
        links = extract_book_links(text2.text)
        #print(links)
        for n in links:
            grab = get_product_info(requests.get('https://books.toscrape.com/catalogue/' + n).text,categories)
            if grab != None:
                #print(grab)
                df = pd.concat([df,pd.DataFrame(columns = grab.keys() , data = [grab.values()])], ignore_index=True)
            # df = pd.concat([df,)])
                #display(pd.DataFrame(grab))
    return df


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a DataFrame.

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[-1]
    'June 03, 19'
    """
    ...


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billions of dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    >>> stats[1][-1] == 'B'
    True
    """
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def get_comments(storyid):
    """
    Returns a DataFrame of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    df = pd.DataFrame(columns = ['id','by','text','parent','time'])

    traverse = []

    page_json = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{storyid}.json').json() 
    page_json

    traverse = traverse + page_json['kids']


    while len(traverse) > 1:
        
        next_id = traverse.pop(0)
        curr_json = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{next_id}.json').json()
        #curr_soup
        if('kids' in curr_json.keys()) & (~('dead' in curr_json.keys())):
            traverse = curr_json['kids'] + traverse
        data = dict({'id':curr_json['id'],'by':curr_json['by'],'text':curr_json['text'],'parent':curr_json['parent'],'time':pd.Timestamp(curr_json['time'],unit='s' )})
        

        df = pd.concat([df, pd.DataFrame(columns = data.keys(), data = [data.values()])], ignore_index=True)
        #print(curr_json['text'])
        #print()
        
        
        
        
    #print(next_id)
    #traverse
    return df
