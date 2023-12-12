import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
PATH_HADATASETS = "/hadatasets/fillipe.silva/"
PAHT_LOCAL_DATASETS = ROOT_DIR + "/datasets/"

VEC_SIZE = 768
RNDN = 42 #random_state
#ALGORITHMS = ['tfidf',  'doc2word', 'bert',    'roberta']
#ALGORITHMS = ['tfidf',	'doc2word',	'bert_imdb', 'roberta_imdb']
#METHODS = ['adjusted_rand_score', 'adjusted_mutual_info_score', 'bagclust', 'han', 'OTclust']

WANDB_API = "d6007776dac6cc8e1b9a985941b906844a13207b"