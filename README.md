# Semantic-parser

## Overview : 

- The code is a set of functions that help you parse [semantic scholar](https://www.semanticscholar.org/paper/DAG-GNN%3A-DAG-Structure-Learning-with-Graph-Neural-Yu-Chen/1d6b8803f6f6b188802275210eb5d7839644a8b5) paper page content. 
Given a paper's id, the main function will return a .json file of all the important information about the paper. This will also serve as the base for the [semantic-vis app's API](https://github.com/AnasAito/Shemantic-vis##).

## Motivation  :

>  A semantic scholar's API already exists, then why make this parser? 
- The Data offered by the existant API is shallow (abstract , title , citations and refs ), it lacks many information such as topics, sorting of citations and refrences and more...
> So how does this work? 
- This tool loads the paper page and then parse all the data about the citations or the paper. Also, you can make some sorting on the citation list and the reference's

## Response Format : 

```yaml
{
 'title': 'Generative Adversarial Nets',
 'corpus_id': 'Corpus ID: 1033682',
 'additional_data': 'Published in NIPS 2014',
 'citation_count': '19,622',
 'citations_overview': {
  'cit_titles': ['Highly Influencial Citations','Background Citations',' Methods Citations', ' Results Citations'],
  'cit_count': ['3,769', '11,284', '6,519', '176']},
 'topics': ['Generative model','Discriminative model', 'Backpropagation', 'Minimax',..],
 'citations': [
 {
   'title': 'Generative Adversarial Learning Towards Fast Weakly Supervised Detection',
   'link': '/paper/Generative-Adversarial-Learning-Towards-Fast-Weakly-Shen-Ji/862b9feff7c5f40736d83bbf10abe32c2702c490',
   'stats': ['40', 'Highly Influenced', 'PDF']}
   ,...]},
 'references': [
 {
   'title': 'Deep Generative Stochastic Networks Trainable by Backprop',
   'link': '/paper/Deep-Generative-Stochastic-Networks-Trainable-by-Bengio-Thibodeau-Laufer/5ffa8bf1bf3e39227be28de4ff6915d3b21eb52d',
   'stats': ['313', 'PDF']
  },...]
  }
```

## Getting started

> script version

- run `python app.py` will run a local server on your machine
- go to `'http://127.0.0.1:5000/parse?paper_id=/paper/Generative-Adversarial-Nets-Goodfellow-Pouget-Abadie/54e325aee6b2d476bbbb88615ac15e251c6e8214'` for example will return a json file containg all the info about [this paper](https://www.semanticscholar.org/paper/Generative-Adversarial-Networks-Goodfellow-Pouget-Abadie/5e9ed7e980634562a329e85ad225d23ea77f3d1f)
- for any paper just go to `'http://127.0.0.1:5000/parse?paper_id=[<paper_id>]'`

> notebook version

- open [`S2 scrapper.ipynb`](https://github.com/AnasAito/Semantic-parser/blob/main/S2%20scrapper.ipynb) there you will find some code example to play with the scrapper and some ideas how to use it .

## Contribution

- Please feel free to contribute to the parser or just suggest additional field to be scrapped !
- or just share with us your use cases .
