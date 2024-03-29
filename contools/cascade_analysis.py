# object for analysing hit_histograms from cascades run using TraverseDispatcher
import numpy as np
import pandas as pd
from contools.process_matrix import Promat
from contools.traverse import Cascade, to_transmission_matrix, TraverseDispatcher

from joblib import Parallel, delayed
from tqdm import tqdm

class Cascade_Analyzer:
    def __init__(self, name, hit_hist, n_init, pairs=[], pairwise=False, skids_in_hit_hist=True, adj_index=None): # changed mg to adj_index for custom/modified adj matrices
        self.hit_hist = hit_hist
        self.name = name
        self.n_init = n_init

        if(skids_in_hit_hist):
            self.adj_index = hit_hist.index
            self.skid_hit_hist = hit_hist

        if(skids_in_hit_hist==False):
            self.adj_index = adj_index
            self.skid_hit_hist = pd.DataFrame(hit_hist, index = self.adj_index) # convert indices to skids

        if(pairwise):
            self.pairs = pairs
            self.hh_inter = self.interlaced_hit_hist()
            self.hh_pairwise = self.average_pairwise_hit_hist()

    def get_hit_hist(self):
        return(self.hit_hist)

    def get_skid_hit_hist(self):
        return(self.skid_hit_hist)

    def get_name(self):
        return(self.name)

    def set_pairs(self, pairs):
        self.pairs = pairs

    def set_hh_inter(self, hh_inter):
        self.hh_inter = hh_inter

    def set_hh_pairwise(self, hh_pairwise):
        self.hh_pairwise = hh_pairwise

    def index_to_skid(self, index):
        return(self.adj_index[index].name)

    def skid_to_index(self, skid):
        index_match = np.where(self.adj_index == skid)[0]
        if(len(index_match)==1):
            return(int(index_match[0]))
        if(len(index_match)!=1):
            print(f'Not one match for skid {skid}!')
            return(False)

    def interlaced_hit_hist(self):

        hit_hist = self.skid_hit_hist.copy()
        skids = hit_hist.index
        skids_pairs, skids_unpaired, skids_nonpaired = Promat.extract_pairs_from_list(skids, self.pairs)
        
        # left_right interlaced order for skid_hit_hist
        pair_order = []
        for i in range(0, len(skids_pairs)):
            pair_order.append(skids_pairs.iloc[i].leftid)
            pair_order.append(skids_pairs.iloc[i].rightid)

        order = pair_order + list(skids_nonpaired.nonpaired)
        interlaced_hit_hist = hit_hist.loc[order, :]

        index_df = pd.DataFrame([['pairs', Promat.get_paired_skids(skid, self.pairs)[0], skid] for skid in pair_order] + [['nonpaired', skid, skid] for skid in list(skids_nonpaired.nonpaired)], 
                                columns = ['pair_status', 'pair_id', 'skid'])
        index = pd.MultiIndex.from_frame(index_df)

        interlaced_hit_hist.index = index
        return(interlaced_hit_hist)

    def average_pairwise_hit_hist(self):

        hit_hist = self.hh_inter.copy()
        hit_hist = hit_hist.groupby('pair_id', axis = 'index').sum()

        order = [x[1] for x in self.hh_inter.index] # pulls just pair_id
        
        # remove duplicates (in pair_ids)
        order_unique = []
        for x in order:
            if (order_unique.count(x) == 0):
                order_unique.append(x)

        # order as before
        hit_hist = hit_hist.loc[order_unique, :]

        # regenerate multiindex
        index = [x[0:2] for x in self.hh_inter.index] # remove skid ids from index

        # remove duplicates (in pair_ids)
        index_unique = []
        for x in index:
            if (index_unique.count(x) == 0):
                index_unique.append(x)

        # add back appropriate multiindex
        index_df = pd.DataFrame(index_unique, columns = ['pair_status', 'pair_id'])
        index_df = pd.MultiIndex.from_frame(index_df)
        hit_hist.index = index_df

        # convert to average (from sum) for paired neurons
        hit_hist.loc['pairs'] = hit_hist.loc['pairs'].values/2

        return(hit_hist)

    def pairwise_threshold(self, threshold, hops, excluded_skids=[], include_source=False, return_pair_ids=False):

        pairs = self.pairs
        hit_hist = self.hh_pairwise

        # identify downstream neurons
        if(include_source):
            ds_bool = np.where((hit_hist.iloc[:, 0:(hops+1)]).sum(axis=1)>=threshold)[0]
        if(include_source==False):
            ds_bool = np.where((hit_hist.iloc[:, 1:(hops+1)]).sum(axis=1)>=threshold)[0]

        # get pair_ids from boolean
        neurons = [x[1] for x in hit_hist.index[ds_bool]]

        # remove user-defined skids
        if(len(excluded_skids)>0):
            neurons = list(np.setdiff1d(neurons, excluded_skids))

        # expand to include left/right neurons (pairwise hit_hist uses only pair_ids, which are left skids and all nonpaired skids)
        if(return_pair_ids==False):
            all_neurons = [Promat.get_paired_skids(neuron, pairs) for neuron in neurons] # expand to include left/right neurons
            all_neurons = [x for sublist in all_neurons for x in sublist] # unlist skids
            return(all_neurons)
            
        if(return_pair_ids):
            return(neurons)

    def cascades_in_celltypes(self, cta, hops, start_hop=1, normalize='visits', pre_counts = None):
        skid_hit_hist = self.skid_hit_hist
        n_init = self.n_init
        hits = []
        for celltype in cta.Celltypes:
            total = skid_hit_hist.loc[np.intersect1d(celltype.get_skids(), skid_hit_hist.index), :].sum(axis=0).iloc[start_hop:hops+1].sum()
            if(normalize=='visits'): total = total/(len(celltype.get_skids())*n_init)
            if(normalize=='skids'): total = total/(len(celltype.get_skids()))
            if(normalize=='pre-skids'): total = total/pre_counts
            hits.append([celltype.get_name(), total])

        data = pd.DataFrame(hits, columns=['neuropil', 'visits_norm'])
        return(data)

    def cascades_in_celltypes_hops(self, cta, hops=None, start_hop=0, normalize='visits', pre_counts = None):

        if(hops==None): hops = len(self.skid_hit_hist.columns)

        skid_hit_hist = self.skid_hit_hist
        n_init = self.n_init
        hits = []
        for celltype in cta.Celltypes:
            total = skid_hit_hist.loc[np.intersect1d(celltype.get_skids(), skid_hit_hist.index), :].sum(axis=0).iloc[start_hop:hops]
            if(normalize=='visits'): total = total/(len(celltype.get_skids())*n_init)
            if(normalize=='skids'): total = total/(len(celltype.get_skids()))
            if(normalize=='pre-skids'): total = total/pre_counts
            hits.append(total)

        data = pd.concat(hits, axis=1)
        return(data)

    @staticmethod
    def run_cascade(start_nodes, cdispatch, disable_tqdm=False):
        return(cdispatch.multistart(start_nodes=start_nodes, disable=disable_tqdm))

    @staticmethod
    def run_cascades_parallel(source_skids_list, source_names, stop_skids, adj, p, max_hops, n_init, simultaneous, pairs=[], pairwise=False, disable_tqdm=True):
        # adj format must be pd.DataFrame with skids for index/columns

        source_indices_list = []
        for skids in source_skids_list:
            indices = np.where([x in skids for x in adj.index])[0]
            source_indices_list.append(indices)

        stop_indices = np.where([x in stop_skids for x in adj.index])[0]

        transition_probs = to_transmission_matrix(adj.values, p)

        cdispatch = TraverseDispatcher(
            Cascade,
            transition_probs,
            stop_nodes = stop_indices,
            max_hops=max_hops+1, # +1 because max_hops includes hop 0
            allow_loops = False,
            n_init=n_init,
            simultaneous=simultaneous,
        )

        job = Parallel(n_jobs=-1)(delayed(Cascade_Analyzer.run_cascade)(source_indices_list[i], cdispatch, disable_tqdm=disable_tqdm) for i in tqdm(range(0, len(source_indices_list))))
        data = [Cascade_Analyzer(name=source_names[i], hit_hist=hit_hist, n_init=n_init, skids_in_hit_hist=False, adj_index=adj.index, pairs=pairs, pairwise=pairwise) for i, hit_hist in enumerate(job)]
        return(data)

    @staticmethod
    def run_single_cascade(name, source_skids, stop_skids, adj, p, max_hops, n_init, simultaneous):

        source_indices = np.where([x in source_skids for x in adj.index])[0]
        stop_indices = np.where([x in stop_skids for x in adj.index])[0]
        transition_probs = to_transmission_matrix(adj.values, p)

        cdispatch = TraverseDispatcher(
            Cascade,
            transition_probs,
            stop_nodes = stop_indices,
            max_hops=max_hops+1, # +1 because max_hops includes hop 0
            allow_loops = False,
            n_init=n_init,
            simultaneous=simultaneous,
        )

        cascade = Cascade_Analyzer.run_cascade(start_nodes = source_indices, cdispatch = cdispatch)
        data = Cascade_Analyzer(name=name, hit_hist=cascade, n_init=n_init, skids_in_hit_hist=False, adj_index=adj.index)
        return(data)

    @staticmethod
    def pairwise_threshold(hh_pairwise, pairs, threshold, hops, excluded_skids=[], include_source=False, return_pair_ids=False):

        hit_hist = hh_pairwise

        # identify downstream neurons
        if(include_source):
            ds_bool = np.where((hit_hist.iloc[:, 0:(hops+1)]).sum(axis=1)>=threshold)[0]
        if(include_source==False):
            ds_bool = np.where((hit_hist.iloc[:, 1:(hops+1)]).sum(axis=1)>=threshold)[0]

        # get pair_ids from boolean
        neurons = [x[1] for x in hit_hist.index[ds_bool]]

        # remove user-defined skids
        if(len(excluded_skids)>0):
            neurons = list(np.setdiff1d(neurons, excluded_skids))

        # expand to include left/right neurons (pairwise hit_hist uses only pair_ids, which are left skids and all nonpaired skids)
        if(return_pair_ids==False):
            all_neurons = [Promat.get_paired_skids(neuron, pairs) for neuron in neurons] # expand to include left/right neurons
            all_neurons = [x for sublist in all_neurons for x in sublist] # unlist skids
            return(all_neurons)
            
        if(return_pair_ids):
            return(neurons)