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

