from bs4 import BeautifulSoup
import requests 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import krovetz

## utils
import urllib.request as libreq
from bs4 import BeautifulSoup
import json
import re, string
import networkx as nx
from nltk.corpus import stopwords
from tqdm import tqdm
from networkx.algorithms import community
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import combinations 

from random import random

ks = krovetz.PyKrovetzStemmer()
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
    ## init 
    data = {}
    cits_list=[]
    refs_list = []
    ## genral info 
    data["title"] = soup.find('h1' , {'data-selenium-selector':'paper-detail-title'}).text
    data['corpus_id'] = soup.find('span' , {'data-selenium-selector':'corpus-id'}).text
    data["additional_data"] = soup.find('span' , {'data-selenium-selector':'year-and-venue'}).text
    
    ## citations
    score_card = soup.find('span' , {'class':'scorecard-stat__headline'})
    if score_card : 
        
        data["citation_count"] = score_card.text.split(" ")[0]
        
        ## citations type
        citations_count = [div.text for div in  soup.find_all('div' , {'class':'scorecard-citation__metadata-item'})]
        citations_title = [div.text for div in soup.find_all('div' , {'class':'scorecard-citation__title'})]
        if len(citations_title)< len(citations_count) :
            citations_title.insert(0,'Highly Influencial Citations')
            
        data['citations_overview'] = {"cit_titles":citations_title , "cit_count" : citations_count }
        
    else : 
        data["citation_count"] = ''
        data['citations_overview'] ={}

    ## paper topics 
    is_topics = (soup.find_all('h4',{'class':'card-sidebar__title'}) is not []) |(soup.find_all('h4',{'class':'card-footer__title'}) is not []) 

    if is_topics : 
        
        data['topics'] = [span.text for span in soup.find_all('span',{'class' :'preview-box__target' })]
    else : data['topics'] = []
    
    ## main citations , refs 
    cards = soup.find_all('div', class_='cl-paper-row citation-list__paper-row')
    citations_cards = cards[:10]
    refs_cards = cards [10:]

    for cit in citations_cards : 
        entry = {}
        entry['title'] = cit.find('div' , class_='cl-paper-title').text
        entry['link'] = cit.find('a')['href']
        
        stats_raw = cit.find('div',class_='cl-paper-controls__stats')#.find_all('div',class_='cl-paper-stat') 
        if stats_raw :
            stats = [div.text for div in stats_raw.find_all('div',class_='cl-paper-stat') ]
            #print(stats)
            entry['stats']=stats
        else : 
            entry['stats']=[]


        cits_list.append(entry)
    for ref in refs_cards : 
        entry = {}
        entry['title'] = ref.find('div' , class_='cl-paper-title').text
        entry['link'] = ref.find('a')['href']
        
        stats_raw = ref.find('div',class_='cl-paper-controls__stats')#.find_all('div',class_='cl-paper-stat') 
        if stats_raw :
            stats = [div.text for div in stats_raw.find_all('div',class_='cl-paper-stat') ]
            #print(stats)
            entry['stats']=stats
        else : 
            entry['stats']=[]


        refs_list.append(entry)
        
            

    data['citations'] = cits_list
    data['references'] = refs_list
    
    ## return data dict 
    return data 

def ravel(list_):return [j for sub in list_ for j in sub]
 
def generate_wcloud(topics_list,max_words=100,stopwords_list =['learning','neural' , 'computer','algorithm','network','Artificial','model']):
    '''
    args : 1d topics list (1 x n)
    output : word cloud image 
    '''
    text = ' '.join(topics_list)
    
    # lower max_font_size, change the maximum number of word and lighten the background:
    stopwords = set(stopwords_list)
    wordcloud = WordCloud(max_font_size=100,stopwords=stopwords, max_words=max_words, background_color="white" ).generate(text)
    print(wordcloud)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def get_semantic_scholar_paper_by_api(paper_id , is_external =False ,citationType='relevance') : 
    
    b_url_from_arxiv = 'https://api.semanticscholar.org/v1/paper/arXiv:'#'hep-ph/0610184'
    b_url = 'https://api.semanticscholar.org/v1/paper/'#'0796f6cd7f0403a854d67d525e9b32af3b277331'


    final_url = is_external*(b_url_from_arxiv+paper_id) + (not is_external )*(b_url+paper_id)
    print(final_url)
    with libreq.urlopen(final_url) as url:
         r = url.read()
    data = json.loads(r)
    #print(data)
    return data        


def load_file(filename):

    with open(filename,'r') as f:
        text = f.readlines()
    return text

def preprocess(line):
    stop_words = set(stopwords.words('english'))
    line = [item.lower() for item in line if not item.lower() in stop_words]
    #print(line)
    stemmed_line = [ks.stem(item) for item in line]
    #print(stemmed_line)
    return stemmed_line

def create_graph(text):
    text = [text.translate(str.maketrans('', '', string.punctuation))]
    word_list = []
    G = nx.Graph()
    #pbar = tqdm(total=len(text))
    for line in text:
        #print(line)
        line = (line.strip()).split()
        line = preprocess(line)
        #print(line)
        for i, word in enumerate(line):
            if i != len(line)-1:
                word_a = word
                word_b = line[i+1]
                #print(word_a,word_b)
                if word_a not in word_list:
                    word_list.append(word_a)
                    
                if word_b not in word_list:
                    word_list.append(word_b)
                    
                if G.has_edge(word_a,word_b):
                    G[word_a][word_b]['weight'] += 1
                    G[word_a][word_b]['distance']  = 1/G[word_a][word_b]['weight'] 
                    #print(G.nodes)
                else:
                    G.add_edge(word_a,word_b, weight = 1)
                    G.add_edge(word_a,word_b, distance = 1)
                    #print(G.nodes)
      #  pbar.update(1)
    #pbar.close()
    return G


def calculate_central_nodes(text_network , max_nodes = -1):

    bc = (nx.betweenness_centrality(text_network,weight='weight'))
    #print(bc)
    
    nx.set_node_attributes(text_network, bc, 'betweenness')
    bc_threshold = sorted(bc.values(), reverse=True)[max_nodes]
    to_keep = [n for n in bc if bc[n] > bc_threshold]
    filtered_network = text_network.subgraph(to_keep)
    return filtered_network
def plot_betweenness_centrality(text_network,max_nodes):
    bc = (nx.betweenness_centrality(text_network,weight='weight'))
    sorted_bc = dict(sorted(bc.items(),reverse=True, key=lambda item: item[1]))
    nodes = list(sorted_bc.keys())[:max_nodes]
    values = list(sorted_bc.values())[:max_nodes]
    fig = px.bar( x=nodes, y=values)
    fig.show()
def plot_degree_centrality(text_network,max_nodes , normalized =True):
    if normalized : 
        bc = (nx.degree_centrality(text_network))
    else : 
        bc = dict(text_network.degree())
    #print(bc)
    sorted_bc = dict(sorted(bc.items(),reverse=True, key=lambda item: item[1]))
    nodes = list(sorted_bc.keys())[:max_nodes]
    values = list(sorted_bc.values())[:max_nodes]
    fig = px.bar( x=nodes, y=values)
    fig.show()
def scatter_centralities(text_network) :
    deg = (nx.degree_centrality(text_network))
    bc = (nx.betweenness_centrality(text_network,weight='weight'))
    values_bc =  list(bc.values())
    values_deg =  list(deg.values())
    labels = list(bc.keys())
   # print(labels)
    fig = px.scatter( x=values_deg, y=values_bc,labels = labels)
    fig.show()
    
    
def create_and_assign_communities(text_network):

    communities_generator = community.girvan_newman(text_network)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = {}
    for community_list in next_level_communities:
        for item in community_list:
            communities[item] = next_level_communities.index(community_list)
    return communities

def draw_final_graph(text_network,communities ,with_size=True ):

    pos = nx.spring_layout(text_network,scale=2)
    color_list = []
    color_map = []
    community_count = max(communities.values())
    for i in range(0,community_count+1):
        color_list.append((random(), random(), random()))
    for node in text_network:
        color_index = communities[node]
        color_map.append(color_list[color_index])
    betweenness = nx.get_node_attributes(text_network,'betweenness')
    betweenness = [x*1000 for x in betweenness.values()]
    if with_size : 
         nx.draw(text_network,with_labels=True,node_size=betweenness,font_size=8,node_color=color_map,edge_cmap=plt.cm.Blues)
    else :  nx.draw(text_network,with_labels=True,font_size=10,node_color=color_map,edge_cmap=plt.cm.Blues)
    plt.draw()
    plt.show()


def get_data(paper_where):

    soup = get_shemantic_paper_html(paper_where)
    data = extract_data(soup)
    print(data)
    corpus_id = ''.join(data['corpus_id'].split(' '))
    data_api = get_semantic_scholar_paper_by_api(corpus_id)
    
    abstract = data_api['abstract']
    topics = data['topics']
    print(topics)
    return topics , abstract 
    
    ## get data 
    
    #return  topics
def make_graph_from_abstarct(abstract):
    '''
    generate graph with weights and betweenss from abstarct text 
    args : abstract string 
    return : nx graph 
    '''
    text_network = create_graph(abstract)
    text_network = calculate_central_nodes(text_network,max_nodes = -1)
    return text_network

def procees_topics(topics_raw):
    '''
    procees topics stings to have same format as full graph nodes
    
    args : raw  topic list from api 
    return : cleaned list of topics 
    '''
    topics_words = ((' '.join(topics_raw)).translate(str.maketrans('', '', string.punctuation))).split(' ')
    top = preprocess(topics_words)
    return top
    
def filter_graph(full_graph,topics):
    '''
    process topics list then filter full graph giben topics nodes and merging edges 
    '''
    graph = nx.Graph()
    return filtred_graph




def get_topic_nodes_in_full_g(topics,graph):
    topics_f = procees_topics(topics)
    return list(set(topics_f).intersection(set(list(graph.nodes))))


def all_pairs_From(list_):
    return list(combinations(list_ , 2)) 
def get_Sp(pair,graph ,topics_f):
    is_edgeable = False
    ## get s path
    path = nx.shortest_path(graph,source=pair[0],target=pair[1], weight='distance')
    ## check if path nodes in topics_list
    non_terminal_nodes = path[1:-1]
    is_edgeable = (list(set(non_terminal_nodes).intersection(set(list(topics_f)))) == []) | (non_terminal_nodes==[])
    return path , is_edgeable
def compute_edge_from_path(path,graph):
    #print(path)
    pairs_ = list(zip(path, path[1:] + path[:1]))[:-1] ## delete last couple that link terminal nodes
    #print(pairs_)
    weights = [graph[pair_[0]][pair_[1]]['weight'] for pair_ in pairs_]
    #print(weights)
    edge_weight=0
    for w in weights : 
        edge_weight+=1/w
 
    return 1/edge_weight , (path[0],path[-1])
    

def topic_graph(abstract,topics):
    ## main 
    edge_list = []
    text_network = create_graph(abstract)
    
    ## only on topic words that are in abtract (maybe we can have null edge nodes in future ...)
    topics_f = get_topic_nodes_in_full_g(topics,text_network)
    
    pairs = all_pairs_From(topics_f)
    
    for pair in pairs : 
        path , is_edgeable = get_Sp(pair,text_network,topics_f)
    
        if is_edgeable : 
    
            w,nodes = compute_edge_from_path(path,text_network)
            edge_list.append((nodes[0], nodes[1], {"weight": w}))
    
    G = nx.Graph(edge_list) 
    return G
        
    
    
