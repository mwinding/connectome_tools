# object for analysing hit_histograms from cascades run using TraverseDispatcher
import numpy as np
import pandas as pd
import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/python_code/connectome_tools/')
from connectome_tools.process_matrix import Promat

class Cascade_Analyzer:
    def __init__(self, hit_hist, adj_index, pairs): # changed mg to adj_index for custom/modified adj matrices
        self.hit_hist = hit_hist
        self.adj_index = adj_index
        self.pairs = pairs
        self.skid_hit_hist = pd.DataFrame(hit_hist, index = self.adj_index) # convert indices to skids

    def get_hit_hist(self):
        return(self.hit_hist)

    def get_skid_hit_hist(self):
        return(self.skid_hit_hist)

    def set_pairs(self, pairs):
        self.pairs = pairs

    def index_to_skid(self, index):
        return(self.adj_index[index].name)

    def skid_to_index(self, skid):
        index_match = np.where(self.adj_index == skid)[0]
        if(len(index_match)==1):
            return(int(index_match[0]))
        if(len(index_match)!=1):
            print(f'Not one match for skid {skid}!')
            return(False)

    def pairwise_threshold_detail(self, threshold, hops, excluded_skids=False):

        neurons = np.where((self.skid_hit_hist.iloc[:, 1:(hops+1)]).sum(axis=1)>threshold)[0]
        neurons = self.skid_hit_hist.index[neurons]

        # remove particular skids if included
        if(excluded_skids!=False): 
            neurons = np.delete(neurons, excluded_skids)

        neurons_pairs, neurons_unpaired, neurons_nonpaired = Promat.extract_pairs_from_list(neurons, self.pairs)
        return(neurons_pairs, neurons_unpaired, neurons_nonpaired)

    def pairwise_threshold(self, threshold, hops, excluded_skids=False):
        neurons_pairs, neurons_unpaired, neurons_nonpaired = Cascade_Analyzer.pairwise_threshold_detail(self, threshold, hops, excluded_skids)
        skids = np.concatenate([neurons_pairs.leftid, neurons_pairs.rightid, neurons_nonpaired.nonpaired])
        return(skids)

class Celltype:
    def __init__(self, name, skids):
        self.name = name
        self.skids = skids

    def get_name(self):
        return(self.name)

    def get_skids(self):
        return(self.skids)

    def downstream_pairwise(self, pairs):
        return(pairs)

    def upstream_pairwise(self, pairs):
        return(pairs)

class Celltype_Analyzer:
    def __init__(self, list_Celltypes, adj=[], skids=[]):
        self.Celltypes = list_Celltypes
        self.celltype_names = [celltype.get_name() for celltype in self.Celltypes]
        self.num = len(list_Celltypes) # how many cell types
        self.known_types = []
        self.known_types_names = []
        #self.mg = mg
        self.adj = adj

        if(len(skids)>0):
            self.skids = skids
        if(len(skids)==0):
            self.skids = list(np.unique([x for sublist in self.Celltypes for x in sublist.get_skids()]))

        self.adj_df = []

    def get_celltype_names(self):
        return self.celltype_names

    def generate_adj(self):
        # adjacency matrix only between assigned cell types
        adj_df = pd.DataFrame(self.adj, index = self.skids, columns = self.skids)
        skids = [skid for celltype in self.Celltypes for skid in celltype.get_skids()]
        adj_df = adj_df.loc[skids, skids]

        # generating multiindex for adjacency matrix df
        index_df = pd.DataFrame([[celltype.get_name(), skid] for celltype in self.Celltypes for skid in celltype.get_skids()], 
                                columns = ['celltype', 'skid'])
        index = pd.MultiIndex.from_frame(index_df)

        # add multiindex to both rows and columns
        adj_df.index = index
        adj_df.columns = index

        self.adj_df = adj_df

    def add_celltype(self, Celltype):
        self.Celltypes = self.Celltypes + Celltype
        self.num += 1
        self.generate_adj()

    def set_known_types(self, list_Celltypes, unknown=True):
        if(unknown==True):
            unknown_skids = np.setdiff1d(self.skids, np.unique([skid for celltype in list_Celltypes for skid in celltype.get_skids()]))
            unknown_type = [Celltype('unknown', unknown_skids)]
            list_Celltypes = list_Celltypes + unknown_type
            
        self.known_types = list_Celltypes
        self.known_types_names = [celltype.get_name() for celltype in list_Celltypes]

    def get_known_types(self):
        return(self.known_types)

    # determine membership similarity (intersection over union) between all pair-wise combinations of celltypes
    def compare_membership(self, sim_type):
        iou_matrix = np.zeros((len(self.Celltypes), len(self.Celltypes)))

        for i in range(len(self.Celltypes)):
            for j in range(len(self.Celltypes)):
                if(len(np.union1d(self.Celltypes[i].skids, self.Celltypes[j].skids)) > 0):
                    if(sim_type=='iou'):
                        intersection = len(np.intersect1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        union = len(np.union1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        calculation = intersection/union
                    
                    if(sim_type=='dice'):
                        intersection = len(np.intersect1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        diff1 = len(np.setdiff1d(self.Celltypes[i].get_skids(), self.Celltypes[j].get_skids()))
                        diff2 = len(np.setdiff1d(self.Celltypes[j].get_skids(), self.Celltypes[i].get_skids()))
                        calculation = intersection*2/(intersection*2 + diff1 + diff2)
                    
                    if(sim_type=='cosine'):
                            unique_skids = list(np.unique(list(self.Celltypes[i].get_skids()) + list(self.Celltypes[j].get_skids())))
                            data = pd.DataFrame(np.zeros(shape=(2, len(unique_skids))), columns = unique_skids, index = [i,j])
                            
                            for k in range(len(data.columns)):
                                if(data.columns[k] in self.Celltypes[i].get_skids()):
                                    data.iloc[0,k] = 1
                                if(data.columns[k] in self.Celltypes[j].get_skids()):
                                    data.iloc[1,k] = 1

                            a = list(data.iloc[0, :])
                            b = list(data.iloc[1, :])

                            dot = np.dot(a, b)
                            norma = np.linalg.norm(a)
                            normb = np.linalg.norm(b)
                            calculation = dot / (norma * normb)

                    iou_matrix[i, j] = calculation

        iou_matrix = pd.DataFrame(iou_matrix, index = [f'{x.get_name()} ({len(x.get_skids())})' for x in self.Celltypes], 
                                            columns = [f'{x.get_name()}' for x in self.Celltypes])

        return(iou_matrix)

    # calculate fraction of neurons in each cell type that have previously known cell type annotations
    def memberships(self, by_celltype=True, raw_num=False): # raw_num=True outputs number of neurons in each category instead of fraction
        fraction_type = np.zeros((len(self.known_types), len(self.Celltypes)))
        for i, knowntype in enumerate(self.known_types):
            for j, celltype in enumerate(self.Celltypes):
                if(by_celltype): # fraction of new cell type in each known category
                    if(raw_num==False):
                        if(len(celltype.get_skids())==0):
                            fraction = 0
                        if(len(celltype.get_skids())>0):
                            fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))/len(celltype.get_skids())
                    if(raw_num==True):
                        fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))
                    fraction_type[i, j] = fraction
                if(by_celltype==False): # fraction of each known category that is in new cell type
                    fraction = len(np.intersect1d(celltype.get_skids(), knowntype.get_skids()))/len(knowntype.get_skids())
                    fraction_type[i, j] = fraction

        fraction_type = pd.DataFrame(fraction_type, index = self.known_types_names, 
                                    columns = [f'{celltype.get_name()} ({len(celltype.get_skids())})' for celltype in self.Celltypes])
        return(fraction_type)

    def connectivtiy(self, celltypes, normalize_pre_num = False, normalize_post_num = False):

        #level0_keys = np.unique(self.adj_df.index.get_level_values(0))
        mat = np.zeros((len(celltypes), len(celltypes)))
        for i, key_i in enumerate(celltypes):
            for j, key_j in enumerate(celltypes):
                if(normalize_pre_num==False & normalize_post_num==False):
                    mat[i, j] = self.adj_df.loc[key_i, key_j].values.sum()
                if(normalize_pre_num==True):
                    mat[i, j] = self.adj_df.loc[key_i, key_j].values.sum()/len(self.adj_df.loc[key_i].index)
                if(normalize_post_num==True):
                    mat[i, j] = self.adj_df.loc[key_i, key_j].values.sum()/len(self.adj_df.loc[key_j].index)
        mat = pd.DataFrame(mat, index = celltypes, columns = celltypes)
        return(mat)



    


