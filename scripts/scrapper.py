from bs4 import BeautifulSoup
import requests

# utils


def get_shemantic_paper_html(where):
    '''
    get page html using filters on citation list

    arguments : paper href 
    output : paper html 


    '''

    base_url = 'https://www.semanticscholar.org'
    filter_ = '?citationRankingModelVersion=v0.2.0-0.01&citedPapersSort=relevance&citedPapersLimit=10&citedPapersOffset=0&sort=is-influential'
    URL = base_url+where+filter_

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    return soup


def extract_data(soup):
    # init
    data = {}
    cits_list = []
    refs_list = []
    # genral info
    data["title"] = soup.find(
        'h1', {'data-selenium-selector': 'paper-detail-title'}).text
    data['corpus_id'] = soup.find(
        'span', {'data-selenium-selector': 'corpus-id'}).text
    data["additional_data"] = soup.find(
        'span', {'data-selenium-selector': 'year-and-venue'}).text
    # citations
    score_card = soup.find('span', {'class': 'scorecard-stat__headline'})
    if score_card:

        data["citation_count"] = score_card.text.split(" ")[0]

        # citations type
        citations_count = [div.text for div in soup.find_all(
            'div', {'class': 'scorecard-citation__metadata-item'})]
        citations_title = [div.text for div in soup.find_all(
            'div', {'class': 'scorecard-citation__title'})]
        if len(citations_title) < len(citations_count):
            citations_title.insert(0, 'Highly Influencial Citations')

        data['citations_overview'] = {
            "cit_titles": citations_title, "cit_count": citations_count}

    else:
        data["citation_count"] = ''
        data['citations_overview'] = {}

    # paper topics
    is_topics = soup.find_all('h4', {'class': 'card-sidebar__title'})
    if is_topics:
        data['topics'] = [span.text for span in soup.find_all(
            'span', {'class': 'preview-box__target'})]
    else:
        data['topics'] = []

    # main citations , refs
    cards = soup.find_all(
        'div', class_='cl-paper-row citation-list__paper-row')
    citations_cards = cards[:10]
    refs_cards = cards[10:]

    for cit in citations_cards:
        entry = {}
        entry['title'] = cit.find('div', class_='cl-paper-title').text
        entry['link'] = cit.find('a')['href']

        # .find_all('div',class_='cl-paper-stat')
        stats_raw = cit.find('div', class_='cl-paper-controls__stats')
        if stats_raw:
            stats = [div.text for div in stats_raw.find_all(
                'div', class_='cl-paper-stat')]
            # print(stats)
            entry['stats'] = stats
        else:
            entry['stats'] = []

        cits_list.append(entry)
    for ref in refs_cards:
        entry = {}
        entry['title'] = ref.find('div', class_='cl-paper-title').text
        entry['link'] = ref.find('a')['href']

        # .find_all('div',class_='cl-paper-stat')
        stats_raw = ref.find('div', class_='cl-paper-controls__stats')
        if stats_raw:
            stats = [div.text for div in stats_raw.find_all(
                'div', class_='cl-paper-stat')]
            # print(stats)
            entry['stats'] = stats
        else:
            entry['stats'] = []

        refs_list.append(entry)

    data['citations'] = cits_list
    data['references'] = refs_list

    # return data dict
    return data


