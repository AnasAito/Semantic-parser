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


from random import random

ks = krovetz.PyKrovetzStemmer()
def get_shemantic_paper_html(where):
    '''
    get page html using filters on citation list
    
    arguments : paper href 
    output : paper html 
    
    
    '''
    
    base_url = 'https://www.semanticscholar.org'
    filter_ = '?citationRankingModelVersion=v0.2.0-0.01&citedPapersSort=relevance&citedPapersLimit=10&citedPapersOffset=0&sort=relevance'
    URL = base_url+where+filter_
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    page = requests.get(URL, headers=headers)
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
    topic_section = soup.find('div',{'data-selenium-selector':'entities-list'})
    is_topics = topic_section is not None
    
    if is_topics : 
        data['topics'] = [span.text for span in topic_section.find_all('span',{'class' :'preview-box__target' })]
    else : 
        data['topics'] = []
    
    ## main citations , refs 
    citations_div = soup.find('div' , {'id':'citing-papers'})
    refs_div = soup.find('div',{'id':'references'})
    citations_cards=[]
    refs_cards = []
    if citations_div is not None : 
         citations_cards = citations_div.find_all('div', class_='cl-paper-row citation-list__paper-row')
  
    if   refs_div is not None :      
         refs_cards = refs_div.find_all('div', class_='cl-paper-row citation-list__paper-row')

    for cit in citations_cards : 
        if cit.find('a') is not None :
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
   # print(refs_cards)
    for ref in refs_cards : 
        #print(ref)
        if ref.find('a') is not None :
        
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
    
## test on root -> 2 citations - > 4 citations (7 papers)


from networkx.readwrite import json_graph
import json
import networkx as nx
from matplotlib import pylab as pl
from itertools import combinations 
def get_data(paper_where,max_leafs=2):
        
    base_url = 'https://www.semanticscholar.org'
    filter_ = '?citationRankingModelVersion=v0.2.0-0.01&citedPapersSort=relevance&citedPapersLimit=10&citedPapersOffset=0&sort=total-citations'
    print(base_url+paper_where+filter_)
    soup = get_shemantic_paper_html(paper_where)
    data = extract_data(soup)
    #print(data)
    corpus_id = ''.join(data['corpus_id'].split(' '))
    data_api = get_semantic_scholar_paper_by_api(corpus_id)
    
    abstract = data_api['abstract']
    topics = data['topics']
    title = data['title']
    citations =  [cit['link'] for cit in data['citations'][:max_leafs]] ## get 2 citation per paper
    #print(topics)
    return topics,title , abstract ,citations
    
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
    

def topic_graph(abstract,topics,title):
    ## main 
    edge_list = []
    text_network = create_graph(abstract)
    topics = topics + title.split(' ')
    #print(topics)
    ## only on topic words that are in abtract (maybe we can have null edge nodes in future ...)
    topics_f = get_topic_nodes_in_full_g(topics,text_network)
    
    pairs = all_pairs_From(topics_f)
    
    for pair in pairs : 
        path , is_edgeable = get_Sp(pair,text_network,topics_f)
    
        if is_edgeable : 
    
            w,nodes = compute_edge_from_path(path,text_network)
            edge_list.append((nodes[0], nodes[1], {"weight": w }))
    
    G = nx.Graph(edge_list) 
    return G
        


def add_attributes(graph):
    degrees_dict = nx.degree_centrality(graph) 
    nodes_comm = create_and_assign_communities(graph)
    ## add attr to nodes
    for node in graph.nodes : 
        graph.nodes[node]['degree'] = degrees_dict[node]
       ## print(nodes_comm[node])
        graph.nodes[node]['community'] = nodes_comm[node]
    ## add attr to edges 
    for edge in graph.edges:
        w = graph[edge[0]][edge[1]]['weight']
        graph[edge[0]][edge[1]]['distance'] = 1/w
    
    return graph

def final_graph_light(graphs_list,title):
    all_nodes = [ ]
    all_edges = []
    for graph_dict in graphs_list:
        
        nodes = list(graph_dict['graph'].nodes)
        edges = list(graph_dict['graph'].edges)
        
        
        all_nodes = list(set(all_nodes+nodes)) 
        all_edges = list(set(all_edges+edges)) 
        print(len(all_edges))
        
    
    final_g = nx.Graph()
    final_g.add_edges_from(all_edges) 
    export_graph({'title':title, 'graph':final_g})
    
    write_title(title)
    return final_g


    

def final_graph_weighted(graphs_list,title):
    all_nodes = [ ]
    all_edges = []
    final_g = nx.Graph()
    
    for graph_dict in graphs_list:
        graph = graph_dict['graph']
        
        for edge in list(graph.edges):
            
            node_a = edge[0]
            node_b = edge[1]
            
            w =graph[node_a][node_b]['weight']
          
           
            if final_g.has_edge(node_a,node_b):
                
                final_g[node_a][node_b]['weight'] += w
                
            else:
             
                final_g.add_edge(node_a,node_b, weight = w)
                
    ## add degree centrality
    final_g_w_degree = add_attributes(final_g)
    ## write to db             
    export_graph({'title':title, 'graph':final_g_w_degree})
    write_title(title)
    return final_g_w_degree 
        
def export_graph(graph_dict):
    data = json_graph.node_link_data(graph_dict['graph'])
    formated_title = graph_dict['title'].replace('/', '')
    path = './graph-ui/src/data/'+formated_title+'.json'
    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    
def write_title(title):
    formated_title = title.replace('/', '')
    with open('./graph-ui/src/data/index.json') as json_file:
        data = json.load(json_file)
       ## print(data['graphs'])
        list_ = data['graphs']
    with open('./graph-ui/src/data/index.json','w') as json_file:
    
        list_ = list_+[formated_title]    
        json.dump({'graphs':list_}, json_file) 
    return 'written'
    
def plot_graph(graph):
    pl.figure()
    nx.draw_networkx(graph)
    pl.show()                
        
##  __main__ from root paper


graphs_list = [ ]
n_iter = 7
exp_title='RL'
print('depth = ' , 0 )
## root paper 
root_url = '/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776'
topics ,title, abstract,citations= get_data(root_url,max_leafs=1000)


print('topics : ',topics)
print('abstract :', abstract )
print('citations : ', citations)

graph = topic_graph(abstract,topics,title)
graphs_list.append({'title':title,'graph':graph})
plot_graph(graph)
print('---------------------------------------')
for i in range(n_iter):
    print('depth = ' , i+1 )
    cit_all = [] 
    for cit_link in citations : 
        topics_ ,title_, abstract_,citations_= get_data(cit_link,max_leafs=1000)
        keywords = topics_+title_.split(' ')
        computable = not (abstract_==None or keywords==[])
        if computable :
            print('topics : ',topics_+title_.split(' '))
            graph = topic_graph(abstract_,topics_,title_)
            plot_graph(graph)
            graphs_list.append({'title':title_,'graph':graph})
            cit_all = cit_all +citations_ 
    citations = cit_all 
    print('leafs count = ',len(citations))
        
    print('---------------------------------------')
        
## export graphs
[export_graph(graphs_dic) for graphs_dic in graphs_list ]

[write_title(graphs_dic['title']) for graphs_dic in graphs_list ]

## final graph
final_g = final_graph_weighted(graphs_list,exp_title+' final graph')
        
 ##  __main__ from db corpus


graphs_list = [ ]

exp_title='UM6P'
path_to_db = './um6p_corpus.xls'
data = pd.read_excel(path_to_db)

for index, row in data.iterrows():
    computable = True
    ## get abstract
   
    if str(row['Abstract'])=='nan':
        
            computable = False
    else : abstract_ = row['Abstract']
    
    ## get title
    if str(row['Article Title'])=='nan':
            title_ = ''
    else : title_= row['Article Title']
    
    ## get topics 
    if str(row['Author Keywords'])=='nan':
        keys = []
    else : keys = row['Author Keywords'].split(';')
        
    if str(row['Keywords Plus'])=='nan':
        keys_p = []
    else : keys_p = row['Keywords Plus'].split(';')   
        
    topics_ =  keys_p +keys
                           
    if topics_ == []:
        computable = False
    
    if computable :   
            print('Paper : ', title_)
            graph = topic_graph(abstract_,topics_,'')
            plot_graph(graph)
            graphs_list.append({'title':title_,'graph':graph})

    print('---------------------------------------')
        
## export graphs
[export_graph(graphs_dic) for graphs_dic in graphs_list ]

[write_title(graphs_dic['title']) for graphs_dic in graphs_list ]

## final graph
final_g = final_graph_weighted(graphs_list,exp_title+' final graph')  


def get_connectivity(graph,node_a,node_b):
    
    path = nx.shortest_path(graph,source=node_a,target=node_b , weight='distance')
    pairs_ = list(zip(path, path[1:] + path[:1]))[:-1] ## delete last couple that link terminal nodes
    #print(pairs_)
    weights = [graph[pair_[0]][pair_[1]]['weight'] for pair_ in pairs_]
    #print(weights)
    edge_weight=0
    for w in weights : 
        edge_weight+=1/w
    return 1/edge_weight , path


def get_similarity(graph,topic,topics):

    conn=[]

    for node in graph.nodes : 
        if node in topics:
            pass

        else :
        
            try : 
                w,path =  get_connectivity(graph,topic,node)
                conn.append(w)
                ##conn_index.append(node)
            except :
                conn.append(0)
    return conn

def get_df(graph,topics):
    dict_values={}
   # topics = ['plant','waste','battery','soil']
    for topic in topics : 
        value_list = get_similarity(graph,topic,topics)
        normed = (np.array(value_list) - min(value_list))/(max(value_list)-min(value_list))
        dict_values[topic] = normed
    filtred = [node for node in graph.nodes if node not in topics]   
    df = pd.DataFrame(data=dict_values).set_index([pd.Index(filtred)])
    return df

def get_clusters(df):
    # Convert DataFrame to matrix
    mat = df.values
    # Using sklearn
    km = KMeans(n_clusters=4)
    km.fit(mat)
    # Get cluster assignment labels
    labels = km.labels_
    # Format results as a DataFrame
    results = pd.DataFrame([df.index,labels]).T
    results = results.rename(columns={0: 'index',1:'cat'})
    return results

def get_graph_logs(log_dict , graphs_list,index,freq):
    dict_ = {}
    new_nodes=[]
    ## get barycenter 
    evo_graph = final_graph_weighted(graphs_list,title='test' , export = False)
    graphs = [evo_graph.subgraph(c).copy() for c in nx.connected_components(evo_graph)]
    bars = [nx.algorithms.distance_measures.barycenter(graph) for graph in graphs]
    lens = [len(graph.nodes) for graph in graphs]
    log_dict = {'bars':bars,'lens':lens}
    dict_['barycenter'] = log_dict
    
    ## get new nodes added 
    new_graphs = graphs_list[index-freq-1:index+1]
    for graph in new_graphs : 
        graph = graph['graph']
        new_nodes.append(list(graph.nodes))
    dict_['new_nodes'] = set(list(chain.from_iterable(new_nodes))) 
    
    ## write data 
    print(dict_)
    #log_dict[index] = dict_
    return evo_graph,dict_
    
        
    
def write_logs(log_list,title_base):
    logs_light=[]
    for index,log in enumerate(log_list) :
        ## write graph to log folder
        data = json_graph.node_link_data(log['graph'])
        formated_title =title_base+'_'+str(index)
        path = './logs/'+formated_title+'.json'
        with open(path, 'w') as outfile:
            json.dump(data, outfile)
        ## write object 
        dict_ = log['logs']
        dict_['title'] = formated_title
        logs_light.append(dict_)
    ## pickle log_light 

    PIK = './logs/'+title_base+".dat"

  
    with open(PIK, "wb") as f:
        pickle.dump(logs_light, f)
   # with open(PIK, "rb") as f:
    #    print pickle.load(f)
        
def get_evo_object(name):  
    # give exp name to rebuild dynamic data (with class graph)
    new_log = []
    path = './logs/'+name+".dat"
    with open(path, "rb") as f:
        index_ =pickle.load(f)
    for log in index_ : 
        dict_ = {}
        ## get logs 
        dict_['logs'] = log
        title = log['title']
        ## get graph 
        path = './logs/'+title+'.json'
        with open(path) as json_file:
            data = json.load(json_file)
            graph = nx.node_link_graph(data)
            dict_['graph'] = graph
        new_log.append(dict_)
    return new_log
    
# __exp__ temporal graph evolution 
##  __main__ from db corpus


graphs_list = [ ]
logger = []
log_dict = {}
exp_title='UM6P'
freq=20
path_to_db = './um6p_crpus_by_time.xls'
data = pd.read_excel(path_to_db)

for index, row in data.iterrows():
    computable = True
    ## get abstract
   
    if str(row['Abstract'])=='nan':
        
            computable = False
    else : abstract_ = row['Abstract']
    
    ## get title
    if str(row['Article Title'])=='nan':
            title_ = ''
    else : title_= row['Article Title']
    
    ## get topics 
    if str(row['Author Keywords'])=='nan':
        keys = []
    else : keys = row['Author Keywords'].split(';')
        
    if str(row['Keywords Plus'])=='nan':
        keys_p = []
    else : keys_p = row['Keywords Plus'].split(';')   
        
    topics_ =  keys_p +keys
                           
    if topics_ == []:
        computable = False
    
    if computable :   
            print('Paper : ', title_)
            graph = topic_graph(abstract_,topics_,'')
            plot_graph(graph)
            graphs_list.append({'title':title_,'graph':graph})
            
            # compute evolution graph : 
            if index%freq==0 or index==len(data)-1 : 
                graph,logs  = get_graph_logs(log_dict , graphs_list,index,freq)
                logger.append({'graph':graph,'logs':logs})   
                
    print('---------------------------------------')
        
## export graphs
#[export_graph(graphs_dic) for graphs_dic in graphs_list ]

#[write_title(graphs_dic['title']) for graphs_dic in graphs_list ]

## final graph
#final_g = final_graph_weighted(graphs_list,exp_title+' final graph')          
    



    
