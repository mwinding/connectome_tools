# %%
import pymaid
import contools
from contools import generate_adjs
import numpy as np
from datetime import datetime

from pymaid_creds import url, name, password, token
from data_settings import data_date, pairs_path
data_date = datetime.today().strftime('%Y-%m-%d')
rm = pymaid.CatmaidInstance(url, token, name, password)

# select neurons to include in adjacency matrices
all_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
all_neurons = list(np.setdiff1d(all_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)

# specify split tags
split_tag = 'mw axon split'
special_split_tags = ['mw axon start', 'mw axon end']
not_split_skids = pymaid.get_skids_by_annotation(['mw unsplittable'])

generate_adjs.adj_split_axons_dendrites(all_neurons, split_tag, special_split_tags, not_split_skids)

# %%
