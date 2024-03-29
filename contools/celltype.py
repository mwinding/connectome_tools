# object for analysing and grouping celltypes

import numpy as np
import pandas as pd
import seaborn as sns
import sys
import pymaid
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools

from upsetplot import plot
from upsetplot import from_contents
from upsetplot import from_memberships
from contools.cascade_analysis import Cascade_Analyzer
from contools.process_matrix import Promat
from contools.process_graph import Analyze_Nx_G

import navis

class Celltype:
    def __init__(self, name, skids, color=None):
        self.name = name
        self.skids = list(np.unique(skids))
        if(color!=None):
            self.color = color

    def get_name(self):
        return(self.name)

    def get_skids(self):
        return(self.skids)

    def get_color(self):
        return(self.color)

    # plots memberships of celltype in a list of other celltypes
    def plot_cell_type_memberships(self, celltypes): # list of Celltype objects
        celltype_colors = [x.get_color() for x in celltypes] + ['tab:gray']

        ct_analyzer = Celltype_Analyzer([self])
        ct_analyzer.set_known_types(celltypes)
        memberships = ct_analyzer.memberships()

        # plot memberships
        ind = np.arange(0, len(ct_analyzer.Celltypes))
        plt.bar(ind, memberships.iloc[0], color=celltype_colors[0])
        bottom = memberships.iloc[0]
        for i in range(1, len(memberships.index)):
            plt.bar(ind, memberships.iloc[i], bottom = bottom, color=celltype_colors[i])
            bottom = bottom + memberships.iloc[i]

    def identify_LNs(self, threshold, summed_adj, aa_adj, input_skids, outputs, exclude, pairs_path, sort = True, use_outputs_in_graph = False):
        pairs = Promat.get_pairs(pairs_path)
        mat = summed_adj.loc[np.intersect1d(summed_adj.index, self.skids), np.intersect1d(summed_adj.index, self.skids)]
        mat = mat.sum(axis=1)
        outputs_in_graph = summed_adj.loc[np.intersect1d(summed_adj.index, self.skids), :]
        outputs_in_graph = outputs_in_graph.sum(axis=1)

        mat_axon = aa_adj.loc[np.intersect1d(aa_adj.index, self.skids), np.intersect1d(aa_adj.index, input_skids)]
        mat_axon = mat_axon.sum(axis=1)

        # convert to % outputs
        skid_percent_output = []
        for skid in self.skids:
            skid_output = 0
            output = sum(outputs.loc[skid, :])
            if(use_outputs_in_graph):
                output = outputs_in_graph.loc[skid]

            if(output != 0):
                if(skid in mat.index):
                    skid_output = skid_output + mat.loc[skid]/output
                if(skid in mat_axon.index):
                    skid_output = skid_output + mat_axon.loc[skid]/output

            skid_percent_output.append([skid, skid_output])

        skid_percent_output = Promat.convert_df_to_pairwise(pd.DataFrame(skid_percent_output, columns=['skid', 'percent_output_intragroup']).set_index('skid'), pairs_path='', pairs=pairs)

        # identify neurons with >={threshold}% output within group (or axoaxonic onto input neurons to group)
        LNs = skid_percent_output.groupby('pair_id').mean()      
        LNs = LNs[np.array([x for sublist in (LNs>=threshold).values for x in sublist])]
        LNs = list(LNs.index) # identify pair_ids of all neurons pairs/nonpaired over threshold
        LNs = [list(skid_percent_output.loc[(slice(None), skid), :].index) for skid in LNs] # pull all left/right pairs or just nonpaired neurons
        LNs = [x[2] for sublist in LNs for x in sublist]
        LNs = list(np.setdiff1d(LNs, exclude)) # don't count neurons flagged as excludes: for example, MBONs/MBINs/RGNs probably shouldn't be LNs
        return(LNs, skid_percent_output)
    
    def identify_in_out_LNs(self, threshold, summed_adj, outputs, inputs, exclude, pairs_path, sort = True):
        pairs = Promat.get_pairs(pairs_path)
        mat_output = summed_adj.loc[:, np.intersect1d(summed_adj.index, self.skids)]
        mat_output = mat_output.sum(axis=1)
        mat_input = summed_adj.loc[np.intersect1d(summed_adj.index, self.skids), :]
        mat_input = mat_input.sum(axis=0)
        
        # convert to % outputs
        skid_percent_in_out = []
        for skid in summed_adj.index:
            skid_output = 0
            output = sum(outputs.loc[skid, :])
            if(output != 0):
                if(skid in mat_output.index):
                    skid_output = skid_output + mat_output.loc[skid]/output

            skid_input = 0
            input_ = sum(inputs.loc[skid, :])
            if(input_ != 0):
                if(skid in mat_input.index):
                    skid_input = skid_input + mat_input.loc[skid]/input_
            
            skid_percent_in_out.append([skid, skid_input, skid_output])

        skid_percent_in_out = Promat.convert_df_to_pairwise(pd.DataFrame(skid_percent_in_out, columns=['skid', 'percent_input_from_group', 'percent_output_to_group']).set_index('skid'), pairs_path='', pairs=pairs)

        # identify neurons with >={threshold}% output within group (or axoaxonic onto input neurons to group)
        LNs = skid_percent_in_out.groupby('pair_id').mean()      
        LNs = LNs[((LNs>=threshold).sum(axis=1)==2).values]
        LNs = list(LNs.index) # identify pair_ids of all neurons pairs/nonpaired over threshold
        LNs = [list(skid_percent_in_out.loc[(slice(None), skid), :].index) for skid in LNs] # pull all left/right pairs or just nonpaired neurons
        LNs = [x[2] for sublist in LNs for x in sublist]
        LNs = list(np.setdiff1d(LNs, exclude)) # don't count neurons flagged as excludes: for example, MBONs/MBINs/RGNs probably shouldn't be LNs
        return(LNs, skid_percent_in_out)

    def plot_morpho(self, figsize, save_path=None, alpha=1, color=None, volume=None, vol_color = (250, 250, 250, .05), azim=-90, elev=-90, dist=6, xlim3d=(-4500, 110000), ylim3d=(-4500, 110000), linewidth=1.5, connectors=False):
        # recommended volume for L1 dataset, 'PS_Neuropil_manual'

        neurons = pymaid.get_neurons(self.skids)

        if(color==None):
            color = self.color

        if(volume!=None):
            neuropil = pymaid.get_volume(volume)
            neuropil.color = vol_color
            fig, ax = navis.plot2d([neurons, neuropil], method='3d_complex', color=color, linewidth=linewidth, connectors=connectors, cn_size=2, alpha=alpha)

        if(volume==None):
            fig, ax = navis.plot2d([neurons], method='3d_complex', color=color, linewidth=linewidth, connectors=connectors, cn_size=2, alpha=alpha)

        ax.azim = azim
        ax.elev = elev
        ax.dist = dist
        ax.set_xlim3d(xlim3d)
        ax.set_ylim3d(ylim3d)

        plt.show()

        if(save_path!=None):
            fig.savefig(f'{save_path}.png', format='png', dpi=300, transparent=True)


class Celltype_Analyzer:
    def __init__(self, list_Celltypes, adj=[]):
        self.Celltypes = list_Celltypes
        self.celltype_names = [celltype.get_name() for celltype in self.Celltypes]
        self.num = len(list_Celltypes) # how many cell types
        self.known_types = []
        self.known_types_names = []
        self.adj = adj
        self.skids = [x for sublist in [celltype.get_skids() for celltype in self.Celltypes] for x in sublist]

    def add_celltype(self, celltype):
        self.Celltypes = self.Celltypes + [celltype]
        self.num += 1
        self.skids = [x for sublist in [celltype.get_skids() for celltype in self.Celltypes] for x in sublist]
        self.celltype_names = [celltype.get_name() for celltype in self.Celltypes]

    def set_known_types(self, list_Celltypes, unknown=True):

        if(list_Celltypes=='default'): data, list_Celltypes = Celltype_Analyzer.default_celltypes()
        if(unknown==True):
            unknown_skids = np.setdiff1d(self.skids, np.unique([skid for celltype in list_Celltypes for skid in celltype.get_skids()]))
            unknown_type = [Celltype('unknown', unknown_skids, 'tab:gray')]
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
    def memberships(self, by_celltype=True, raw_num=False, raw_num_col=True): # raw_num=True outputs number of neurons in each category instead of fraction
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
        
        if(raw_num_col):
            fraction_type = pd.DataFrame(fraction_type, index = self.known_types_names, 
                                    columns = [f'{celltype.get_name()} ({len(celltype.get_skids())})' for celltype in self.Celltypes])
        if(raw_num_col==False):
            fraction_type = pd.DataFrame(fraction_type, index = self.known_types_names, 
                                    columns = [celltype.get_name() for celltype in self.Celltypes])
        
        if(raw_num==True):
            fraction_type = fraction_type.astype(int)
        
        return(fraction_type)

    def plot_memberships(self, path, figsize, rotated_labels = True, raw_num = False, memberships=None, ylim=None, celltype_colors=None):
        if(type(memberships)!=pd.DataFrame):
            memberships = self.memberships(raw_num=raw_num)

        if(celltype_colors==None):
            celltype_colors = [x.get_color() for x in self.get_known_types()]

        # plot memberships
        ind = [cell_type.get_name() for cell_type in self.Celltypes]
        f, ax = plt.subplots(figsize=figsize)
        plt.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
        bottom = memberships.iloc[0, :]
        for i in range(1, len(memberships.index)):
            plt.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
            bottom = bottom + memberships.iloc[i, :]

        if(rotated_labels):
            plt.xticks(rotation=45, ha='right')
        if(ylim!=None):
            plt.ylim(ylim[0], ylim[1])
        plt.savefig(path, format='pdf', bbox_inches='tight')

    def connectivity(self, adj, use_stored_adj=None, normalize_pre_num = False, normalize_post_num = False):

        if(use_stored_adj==True):
            adj_df = self.adj
        else:
            adj_df = adj

        celltypes = [x.get_skids() for x in self.Celltypes]
        celltype_names = [x.get_name() for x in self.Celltypes]
        mat = np.zeros((len(celltypes), len(celltypes)))
        for i, key_i in enumerate(celltypes):
            for j, key_j in enumerate(celltypes):
                if(normalize_pre_num==False & normalize_post_num==False):
                    mat[i, j] = adj_df.loc[np.intersect1d(adj_df.index, key_i), np.intersect1d(adj_df.index, key_j)].values.sum()
                if(normalize_pre_num==True):
                    mat[i, j] = adj_df.loc[np.intersect1d(adj_df.index, key_i), np.intersect1d(adj_df.index, key_j)].values.sum()/len(np.intersect1d(adj_df.index, key_i))
                if(normalize_post_num==True):
                    mat[i, j] = adj_df.loc[np.intersect1d(adj_df.index, key_i), np.intersect1d(adj_df.index, key_j)].values.sum()/len(np.intersect1d(adj_df.index, key_j))
        mat = pd.DataFrame(mat, index = celltype_names, columns = celltype_names)
        return(mat)

    # determine connectivity probability between groups of neurons (using self.Celltypes)
    # must provide paired edge list from Promat.pull_edges(pairs_combined=True) and pairs from Promat.get_pairs()
    def connection_prob(self, edges, pairs=[], pairs_combined=True):

        celltypes = [x.get_skids() for x in self.Celltypes]
        celltype_names = [x.get_name() for x in self.Celltypes]
        mat = np.zeros((len(celltypes), len(celltypes)))


        if(pairs_combined):
            graph = Analyze_Nx_G(edges=edges, graph_type='directed', split_pairs=False)
            for i, celltype1 in enumerate(celltypes):
                for j, celltype2 in enumerate(celltypes):
                    connection = []
                    celltype1 = Promat.extract_pairs_from_list(celltype1, pairs, return_pair_ids=True)
                    celltype2 = Promat.extract_pairs_from_list(celltype2, pairs, return_pair_ids=True)

                    for skid1 in celltype1:
                        for skid2 in celltype2:
                            if((skid1, skid2) in graph.G.edges): connection.append(1)
                            if((skid1, skid2) not in graph.G.edges): connection.append(0)

                    mat[i, j] = sum(connection)/len(connection)

        if(pairs_combined==False):
            graph = Analyze_Nx_G(edges=edges, graph_type='directed', split_pairs=True)
            for i, celltype1 in enumerate(celltypes):
                for j, celltype2 in enumerate(celltypes):
                    connection = []

                    for skid1 in celltype1:
                        for skid2 in celltype2:
                            if((skid1, skid2) in graph.G.edges): connection.append(1)
                            if((skid1, skid2) not in graph.G.edges): connection.append(0)

                    mat[i, j] = sum(connection)/len(connection)
        
        df = pd.DataFrame(mat, columns = celltype_names, index = celltype_names)
        return(df)

    def upset_members(self, threshold=0, path=None, plot_upset=False, show_counts_bool=True, exclude_singletons_from_threshold=False, threshold_dual_cats=None, exclude_skids=None):

        celltypes = self.Celltypes

        contents = {} # empty dictionary
        for celltype in celltypes:
            name = celltype.get_name()
            contents[name] = celltype.get_skids()

        data = from_contents(contents)

        # identify indices of set intersection between all data and exclude_skids
        if(exclude_skids!=None):
            ind_dict = dict((k,i) for i,k in enumerate(data.id.values))
            inter = set(ind_dict).intersection(exclude_skids)
            indices = [ind_dict[x] for x in inter]
            data = data.iloc[np.setdiff1d(range(0, len(data)), indices)]

        unique_indices = np.unique(data.index)
        cat_types = [Celltype(' and '.join([data.index.names[i] for i, value in enumerate(index) if value==True]), 
                    list(data.loc[index].id)) for index in unique_indices]

        # apply threshold to all category types
        if(exclude_singletons_from_threshold==False):
            cat_bool = [len(x.get_skids())>=threshold for x in cat_types]
        
        # allows categories with no intersection ('singletons') to dodge the threshold
        if((exclude_singletons_from_threshold==True) & (threshold_dual_cats==None)): 
            cat_bool = [(((len(x.get_skids())>=threshold) | (' and ' not in x.get_name()))) for x in cat_types]

        # allows categories with no intersection ('singletons') to dodge the threshold and additional threshold for dual combos
        if((exclude_singletons_from_threshold==True) & (threshold_dual_cats!=None)): 
            cat_bool = [(((len(x.get_skids())>=threshold) | (' and ' not in x.get_name())) | (len(x.get_skids())>=threshold_dual_cats) & (x.get_name().count('+')<2)) for x in cat_types]

        cats_selected = list(np.array(cat_types)[cat_bool])
        skids_selected = [x for sublist in [cat.get_skids() for cat in cats_selected] for x in sublist]

        # identify indices of set intersection between all data and skids_selected
        ind_dict = dict((k,i) for i,k in enumerate(data.id.values))
        inter = set(ind_dict).intersection(skids_selected)
        indices = [ind_dict[x] for x in inter]

        data = data.iloc[indices]

        # identify skids that weren't plotting in upset plot (based on plotting threshold)
        all_skids = [x for sublist in [cat.get_skids() for cat in cat_types] for x in sublist]
        skids_excluded = list(np.setdiff1d(all_skids, skids_selected))

        if(plot_upset):
            if(show_counts_bool):
                fg = plot(data, sort_categories_by = None, show_counts='%d')
            else: 
                fg = plot(data, sort_categories_by = None)

            if(threshold_dual_cats==None):
                plt.savefig(f'{path}_excluded{len(skids_excluded)}_threshold{threshold}.pdf', bbox_inches='tight')
            if(threshold_dual_cats!=None):
                plt.savefig(f'{path}_excluded{len(skids_excluded)}_threshold{threshold}_dual-threshold{threshold_dual_cats}.pdf', bbox_inches='tight')

        return (cat_types, cats_selected, skids_excluded)

    # work on this one later
    def plot_morphos(self, figsize, save_path=None, alpha=1, volume=None, vol_color = (250, 250, 250, .05), azim=-90, elev=-90, dist=6, xlim3d=(-4500, 110000), ylim3d=(-4500, 110000), linewidth=1.5, connectors=False):
        # recommended volume for L1 dataset, 'PS_Neuropil_manual'

        neurons = pymaid.get_neurons(self.skids)

        if(volume!=None):
            neuropil = pymaid.get_volume(volume)
            neuropil.color = vol_color
            fig, ax = navis.plot2d([neurons, neuropil], method='3d_complex', color=color, linewidth=linewidth, connectors=connectors, cn_size=2, alpha=alpha)

        if(volume==None):
            fig, ax = navis.plot2d([neurons], method='3d_complex', color=color, linewidth=linewidth, connectors=connectors, cn_size=2, alpha=alpha)

        ax.azim = azim
        ax.elev = elev
        ax.dist = dist
        ax.set_xlim3d(xlim3d)
        ax.set_ylim3d(ylim3d)

        plt.show()

    @staticmethod
    def get_skids_from_meta_meta_annotation(meta_meta, split=False, return_celltypes=False):
        meta_annots = pymaid.get_annotated(meta_meta).name
        annot_list = [list(pymaid.get_annotated(meta).name) for meta in meta_annots]
        skids = [list(pymaid.get_skids_by_annotation(annots)) for annots in annot_list]
        if(split==False):
            skids = [x for sublist in skids for x in sublist]
            return(skids)
        if(split==True):
            if(return_celltypes==True):
                celltypes = [Celltype(meta_annots[i], skids[i]) for i in range(len(meta_annots))]
                return(celltypes)
            if(return_celltypes==False):
                return(skids, meta_annots)

    @staticmethod
    def get_skids_from_meta_annotation(meta, split=False, unique=True, return_celltypes=False):
        annot_list = pymaid.get_annotated(meta).name
        skids = [list(pymaid.get_skids_by_annotation(annots)) for annots in annot_list]
        if(split==False):
            skids = [x for sublist in skids for x in sublist]
            if(unique==True):
                skids = list(np.unique(skids))
            return(skids)
        if(split==True):
            if(return_celltypes==True):
                celltypes = [Celltype(annot_list[i], skids[i]) for i in range(len(annot_list))]
                return(celltypes)
            if(return_celltypes==False):
                return(skids, annot_list)
    
    @staticmethod
    def default_celltypes(exclude = []):
        priority_list = pymaid.get_annotated('mw brain simple priorities').name
        priority_skids = [Celltype_Analyzer.get_skids_from_meta_meta_annotation(priority) for priority in priority_list]

        # made the priority groups exclusive by removing neurons from lists that also in higher priority
        override = priority_skids[0]
        priority_skids_unique = [priority_skids[0]]
        for i in range(1, len(priority_skids)):
            skids_temp = list(np.setdiff1d(priority_skids[i], override))
            priority_skids_unique.append(skids_temp)
            override = override + skids_temp

        # take all 'mw brain simple groups' skids (under 'mw brain simple priorities' meta-annotation)
        #   and remove skids that aren't in the appropriate priority_skids_unique level
        priority_skid_groups = [list(pymaid.get_annotated(meta).name) for meta in priority_list]

        skid_groups = []
        for i in range(0, len(priority_skid_groups)):
            group = []
            for j in range(0, len(priority_skid_groups[i])):
                skids_temp = pymaid.get_skids_by_annotation(pymaid.get_annotated(priority_skid_groups[i][j]).name)
                skids_temp = list(np.intersect1d(skids_temp, priority_skids_unique[i])) # make sure skid in subgroup is set in correct priority list
                
                # remove neurons in optional "exclude" list
                if(len(exclude)>0):
                    skids_temp = list(np.setdiff1d(skids_temp, exclude))

                group.append(skids_temp)
            skid_groups.append(group)

        # test skid counts for each group
        #[len(x) for sublist in skid_groups for x in sublist]

        # make list of lists of skids + their associated names
        skid_groups = [x for sublist in skid_groups for x in sublist]
        names = [list(pymaid.get_annotated(x).name) for x in priority_list]
        names = [x for sublist in names for x in sublist]
        names = [x.replace('mw brain ', '') for x in names]
        
        # identify colors
        colors = list(pymaid.get_annotated('mw brain simple colors').name)

        # official order; note that it will have to change if any new groups are added
        official_order = ['sensories', 'PNs', 'ascendings', 'PNs-somato', 'LNs', 'LHNs', 'FFNs', 'MBINs', 'KCs', 'MBONs', 'MB-FBNs', 'CNs', 'pre-dSEZs', 'pre-dVNCs', 'RGNs', 'dSEZs', 'dVNCs']
        colors_names = [x.name.values[0] for x in list(map(pymaid.get_annotated, colors))] # use order of colors annotation for now
        if(len(official_order)!=len(colors_names)):
            print('warning: issue with annotations! Check "official_order" in Celltype_Analyzer.default_celltypes()')
            
        # ordered properly and linked to colors
        groups_sort = [np.where(x==np.array(official_order))[0][0] for x in names]
        names = [element for _, element in sorted(zip(groups_sort, names))]
        skid_groups = [element for _, element in sorted(zip(groups_sort, skid_groups))]

        color_sort = [np.where(x.replace('mw brain ', '')==np.array(official_order))[0][0] for x in colors_names]
        colors = [element for _, element in sorted(zip(color_sort, colors))]

        data = pd.DataFrame(zip(names, skid_groups, colors), columns = ['name', 'skids', 'color'])
        celltype_objs = list(map(lambda x: Celltype(*x), zip(names, skid_groups, colors)))

        return(data, celltype_objs)


    @staticmethod
    def layer_id(layers, layer_names, celltype_skids):
        max_layers = max([len(layer) for layer in layers])

        mat_neurons = np.zeros(shape = (len(layers), max_layers))
        mat_neuron_skids = pd.DataFrame()
        for i in range(0,len(layers)):
            skids = []
            for j in range(0,len(layers[i])):
                neurons = np.intersect1d(layers[i][j], celltype_skids)
                count = len(neurons)

                mat_neurons[i, j] = count
                skids.append(neurons)
            
            if(len(skids) != max_layers):
                skids = skids + [[]]*(max_layers-len(skids)) # make sure each column has same num elements

            mat_neuron_skids[layer_names[i]] = skids

        id_layers = pd.DataFrame(mat_neurons, index = layer_names, columns = [f'Layer {i+1}' for i in range(0,max_layers)])
        id_layers_skids = mat_neuron_skids

        return(id_layers, id_layers_skids)

    @staticmethod
    def plot_layer_types(layer_types, layer_names, layer_colors, layer_vmax, pair_ids, figsize, save_path, threshold, hops):

        col = layer_colors

        pair_list = []
        for pair in pair_ids:
            mat = np.zeros(shape=(len(layer_types), len(layer_types[0].columns)))
            for i, layer_type in enumerate(layer_types):
                mat[i, :] = layer_type.loc[pair]

            pair_list.append(mat)

        # loop through pairs to plot
        for i, pair in enumerate(pair_list):

            data = pd.DataFrame(pair, index = layer_names)
            mask_list = []
            for i_iter in range(0, len(data.index)):
                mask = np.full((len(data.index),len(data.columns)), True, dtype=bool)
                mask[i_iter, :] = [False]*len(data.columns)
                mask_list.append(mask)

            fig, axs = plt.subplots(
                1, 1, figsize=figsize
            )
            for j, mask in enumerate(mask_list):
                vmax = layer_vmax[j]
                ax = axs
                annotations = data.astype(int).astype(str)
                annotations[annotations=='0']=''
                sns.heatmap(data, annot = annotations, fmt = 's', mask = mask, cmap=col[j], vmax = vmax, cbar=False, ax = ax)

            plt.savefig(f'{save_path}hops{hops}_{i}_{pair_ids[i]}_Threshold-{threshold}_individual-path.pdf', bbox_inches='tight')

    # use df from find_all_partners_hemispheres() if simple=FALSE
    # use df from find_all_partners() if simple=TRUE
    @staticmethod
    def chromosome_plot(df, path, celltypes, plot_type='raw_norm', simple=False, spacer_num=1, col_width = 0.2, col_height=1): 
        df = df.set_index('source_pairid')

        if(simple==False):
            upstream_ct = []
            downstream_ct = []
            for pairid in df.index:
                us_ipsi = df.loc[pairid, 'upstream-ipsi']
                us_contra = df.loc[pairid, 'upstream-contra']
                ds_ipsi = df.loc[pairid, 'downstream-ipsi']
                ds_contra = df.loc[pairid, 'downstream-contra']

                upstream_ct.append(Celltype(f'{pairid}_upstream-ipsi', us_ipsi))
                upstream_ct.append(Celltype(f'{pairid}_upstream-contra', us_contra))
                j = 0
                while(j<spacer_num): # add appropriate number of spacers
                    upstream_ct.append(Celltype(f'{pairid}-spacer-{j}', [])) # add these blank columns for formatting purposes only
                    j+=1

                downstream_ct.append(Celltype(f'{pairid}_downstream-ipsi', ds_ipsi))
                downstream_ct.append(Celltype(f'{pairid}_downstream-contra', ds_contra))
                j = 0
                while(j<spacer_num): # add appropriate number of spacers
                    downstream_ct.append(Celltype(f'{pairid}-spacer-{j}', [])) # add these blank columns for formatting purposes only
                    j+=1

            upstream_ct = Celltype_Analyzer(upstream_ct)
            downstream_ct = Celltype_Analyzer(downstream_ct)

            upstream_ct.set_known_types(celltypes)
            downstream_ct.set_known_types(celltypes)

            if(plot_type=='simple_fraction'):
                upstream_ct.plot_memberships(path, figsize=(col_width*len(upstream_ct.Celltypes),col_height), ylim=(0,1))
                downstream_ct.plot_memberships(path, figsize=(col_width*len(downstream_ct.Celltypes),col_height), ylim=(0,1))
            
            if(plot_type=='raw'):
                upstream_memberships = upstream_ct.memberships(raw_num=True)
                downstream_memberships = downstream_ct.memberships(raw_num=True)
                upstream_ct.plot_memberships(path=path, figsize=(col_width*len(upstream_ct.Celltypes),col_height), memberships=upstream_memberships, ylim=(0,1))
                downstream_ct.plot_memberships(path=path, figsize=(col_width*len(downstream_ct.Celltypes),col_height), memberships=downstream_memberships, ylim=(0,1))
            
            if(plot_type=='raw_norm'):
                upstream_memberships = upstream_ct.memberships(raw_num=True)
                downstream_memberships = downstream_ct.memberships(raw_num=True)

                for i in range(0, len(upstream_memberships.columns), 2+spacer_num):
                    col_name_us = upstream_memberships.columns[i]
                    col_name2_us = upstream_memberships.columns[i+1]
                    col_name_ds = downstream_memberships.columns[i]
                    col_name2_ds = downstream_memberships.columns[i+1]

                    sum_us = sum(upstream_memberships.loc[:, col_name_us]) + sum(upstream_memberships.loc[:, col_name2_us])
                    sum_ds = sum(downstream_memberships.loc[:, col_name_ds]) + sum(downstream_memberships.loc[:, col_name2_ds])

                    upstream_memberships.loc[:, col_name_us] = upstream_memberships.loc[:, col_name_us]/sum_us
                    upstream_memberships.loc[:, col_name2_us] = upstream_memberships.loc[:, col_name2_us]/sum_us
                    downstream_memberships.loc[:, col_name_ds] = downstream_memberships.loc[:, col_name_ds]/sum_ds
                    downstream_memberships.loc[:, col_name2_ds] = downstream_memberships.loc[:, col_name2_ds]/sum_ds

                upstream_ct.plot_memberships(path=path + '-upstream.pdf', figsize=(col_width*len(upstream_ct.Celltypes),col_height), memberships=upstream_memberships, ylim=(0,1))
                downstream_ct.plot_memberships(path=path + '-downstream.pdf', figsize=(col_width*len(downstream_ct.Celltypes),col_height), memberships=downstream_memberships, ylim=(0,1))

        if(simple==True):
            upstream_ct = []
            downstream_ct = []
            for pairid in df.index:
                us = df.loc[pairid, 'upstream']
                ds = df.loc[pairid, 'downstream']

                upstream_ct.append(Celltype(f'{pairid}-upstream', us))
                j = 0
                while(j<spacer_num): # add appropriate number of spacers
                    upstream_ct.append(Celltype(f'{pairid}-spacer-{j}', [])) # add these blank columns for formatting purposes only
                    j+=1

                downstream_ct.append(Celltype(f'{pairid}-downstream', ds))
                j = 0
                while(j<spacer_num): # add appropriate number of spacers
                    downstream_ct.append(Celltype(f'{pairid}-spacer-{j}', [])) # add these blank columns for formatting purposes only
                    j+=1

            upstream_ct = Celltype_Analyzer(upstream_ct)
            downstream_ct = Celltype_Analyzer(downstream_ct)

            upstream_ct.set_known_types(celltypes)
            downstream_ct.set_known_types(celltypes)
            upstream_ct.plot_memberships(path=path + '-upstream.pdf', figsize=(col_width*len(upstream_ct.Celltypes),col_height), ylim=(0,1))
            downstream_ct.plot_memberships(path=path + '-downstream.pdf', figsize=(col_width*len(downstream_ct.Celltypes),col_height), ylim=(0,1))

    @staticmethod
    def plot_marginal_cell_type_cluster(size, particular_cell_type, particular_color, cluster_level, path, all_celltypes=None):

        # all cell types plot data
        if(all_celltypes==None):
            _, all_celltypes = Celltype_Analyzer.default_celltypes()
        
        all_neurons = pymaid.get_skids_by_annotation(['mw brain and inputs', 'mw brain accessory neurons'])
        remove_neurons = pymaid.get_skids_by_annotation(['mw brain very incomplete', 'mw partially differentiated', 'mw motor'])
        all_neurons = list(np.setdiff1d(all_neurons, remove_neurons)) # remove neurons that are incomplete or partially differentiated (as well as SEZ motor neurons)
        clusters = Analyze_Cluster(cluster_lvl=cluster_level, meta_data_path = 'data/graphs/meta_data.csv', skids=all_neurons)

        #all_clusters = [Celltype(lvl.cluster_df.cluster[i], lvl.cluster_df.skids[i]) for i in range(0, len(lvl.clusters))]
        cluster_analyze = clusters.cluster_cta

        cluster_analyze.set_known_types(all_celltypes)
        celltype_colors = [x.get_color() for x in cluster_analyze.get_known_types()]
        all_memberships = cluster_analyze.memberships()
        all_memberships = all_memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
        celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs
        
        # particular cell type data
        cluster_analyze.set_known_types([particular_cell_type])
        membership = cluster_analyze.memberships()

        # plot
        fig = plt.figure(figsize=size) 
        fig.subplots_adjust(hspace=0.1)
        gs = GridSpec(4, 1)

        ax = fig.add_subplot(gs[0:3, 0])
        ind = np.arange(0, len(cluster_analyze.Celltypes))
        ax.bar(ind, membership.iloc[0, :], color=particular_color)
        ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]), title=particular_cell_type.get_name())

        ax = fig.add_subplot(gs[3, 0])
        ind = np.arange(0, len(cluster_analyze.Celltypes))
        ax.bar(ind, all_memberships.iloc[0, :], color=celltype_colors[0])
        bottom = all_memberships.iloc[0, :]
        for i in range(1, len(all_memberships.index)):
            plt.bar(ind, all_memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
            bottom = bottom + all_memberships.iloc[i, :]
        ax.set(xlim = (-1, len(ind)), ylim=(0,1), xticks=([]), yticks=([]))
        ax.axis('off')
        ax.axis('off')

        plt.savefig(path, format='pdf', bbox_inches='tight')

def plot_celltype(path, pairids, n_rows, n_cols, celltypes, pairs_path, plot_pairs=True, connectors=False, cn_size=0.25, color=None, names=False, plot_padding=[0,0]):

    pairs = Promat.get_pairs(pairs_path)
    # pull specific cell type identities
    celltype_ct = [Celltype(f'{pairid}-ipsi-bi', Promat.get_paired_skids(pairid, pairs)) for pairid in pairids]
    celltype_ct = Celltype_Analyzer(celltype_ct)
    celltype_ct.set_known_types(celltypes)
    members = celltype_ct.memberships()

    # link identities to official celltype colors 
    celltype_identities = [np.where(members.iloc[:, i]==1.0)[0][0] for i in range(0, len(members.columns))]
    if(plot_pairs):
        celltype_ct = [Celltype(celltypes[celltype_identities[i]].name.replace('s', ''), Promat.get_paired_skids(pairid, pairs), celltypes[celltype_identities[i]].color) if celltype_identities[i]<17 else Celltype(f'{pairid}', Promat.get_paired_skids(pairid, pairs), '#7F7F7F') for i, pairid in enumerate(pairids)]
    if(plot_pairs==False):
        celltype_ct = [Celltype(celltypes[celltype_identities[i]].name.replace('s', ''), pairid, celltypes[celltype_identities[i]].color) if celltype_identities[i]<17 else Celltype('Other', pairid, '#7F7F7F') for i, pairid in enumerate(pairids)]

    # plot neuron morphologies
    neuropil = pymaid.get_volume('PS_Neuropil_manual')
    neuropil.color = (250, 250, 250, .05)

    n_rows = n_rows
    n_cols = n_cols
    alpha = 1

    fig = plt.figure(figsize=(n_cols*2, n_rows*2))
    gs = plt.GridSpec(n_rows, n_cols, figure=fig, wspace=plot_padding[0], hspace=plot_padding[1])
    axs = np.empty((n_rows, n_cols), dtype=object)

    for i, skids in enumerate([x.skids for x in celltype_ct]):
        if(color!=None):
            col = color
        else: col = celltype_ct[i].color
        neurons = pymaid.get_neurons(skids)

        inds = np.unravel_index(i, shape=(n_rows, n_cols))
        ax = fig.add_subplot(gs[inds], projection="3d")
        axs[inds] = ax
        navis.plot2d(x=[neurons, neuropil], connectors=connectors, cn_size=cn_size, color=col, alpha=alpha, ax=ax, method='3d_complex')

        ax.azim = -90
        ax.elev = -90
        ax.dist = 6
        ax.set_xlim3d((-4500, 110000))
        ax.set_ylim3d((-4500, 110000))
        if(names):
            ax.text(x=(ax.get_xlim()[0] + ax.get_xlim()[1])/2 - ax.get_xlim()[1]*0.05, y=ax.get_ylim()[1]*4/5, z=0, 
                    s=celltype_ct[i].name, transform=ax.transData, color=col, alpha=1)

    fig.savefig(f'{path}.png', format='png', dpi=300, transparent=True)


class Analyze_Cluster():

    # cluster_lvl should be integer between 0 and max levels of cluster hierarchy
    # meta_data_path is the path to a meta_data file in 'data/graphs/'; contains cluster and sort information
    def __init__(self, cluster_lvl, meta_data_path, skids, sort='signal_flow'): # default is skids = pymaid.get_skids_by_annotation('mw brain paper clustered neurons')
                                                                            # meta_data_path = 'data/graphs/meta_data.csv'

        self.meta_data = pd.read_csv(meta_data_path, index_col = 0, header = 0) # load meta_data file
        self.skids = skids

        # determine where neurons are in the signal from sensory -> descending neurons
        # determined using iterative random walks
        self.cluster_order, self.cluster_df = self.cluster_order(cluster_lvl=cluster_lvl, sort=sort)
        self.cluster_cta = Celltype_Analyzer([Celltype(self.cluster_order[i], skids) for i, skids in enumerate(list(self.cluster_df.skids))])

    def cluster_order(self, cluster_lvl, sort='walk_sort'):

        brain_clustered = self.skids

        meta_data_df = self.meta_data.copy()
        meta_data_df['skid']=meta_data_df.index

        cluster_df = pd.DataFrame(list(meta_data_df.groupby(f'dc_level_{cluster_lvl}_n_components=10_min_split=32')['skid']), columns=['cluster', 'skids'])
        cluster_df['skids'] = [x.values for x in cluster_df.skids]
        cluster_df[f'sum_{sort}'] = [np.nanmean(x[1].values) for x in list(meta_data_df.groupby(f'dc_level_{cluster_lvl}_n_components=10_min_split=32')[f'sum_{sort}'])]
        
        # sort from input to output (signal-flow: [X, -X], walk-sort: [0,1])
        if(sort=='signal_flow'):
            cluster_df.sort_values(by=f'sum_{sort}', ascending=False, inplace=True)
        if(sort=='walk_sort'):
            cluster_df.sort_values(by=f'sum_{sort}', ascending=True, inplace=True)

        cluster_df.reset_index(inplace=True, drop=True)

        # returns cluster order and clusters dataframe (with order, skids, sort values)
        return(list(cluster_df.cluster), cluster_df) 

    def ff_fb_cascades(self, adj, p, max_hops, n_init):

        skids_list = list(self.cluster_df.skids)
        source_names = list(self.cluster_df.cluster)
        stop_skids = []
        simultaneous = True
        hit_hists_list = Cascade_Analyzer.run_cascades_parallel(source_skids_list = skids_list, source_names = source_names, stop_skids=stop_skids,
                                                                    adj=adj, p=p, max_hops=max_hops, n_init=n_init, simultaneous=simultaneous)
        return(hit_hists_list)
        
    def all_ff_fb_df(self, cascs_list, normalize='visits'):

        rows = []
        for i, casc_analyzer in enumerate(cascs_list):
            precounts = len(self.cluster_df.skids[i])
            casc_row = casc_analyzer.cascades_in_celltypes(cta=self.cluster_cta, hops=4, start_hop=0, normalize=normalize, pre_counts=precounts)
            rows.append(casc_row)

        ff_fb_df = pd.concat(rows, axis=1)
        ff_fb_df.drop(columns='neuropil', inplace=True)
        ff_fb_df.columns = self.cluster_order
        ff_fb_df.index = self.cluster_order
        return(ff_fb_df)

    def plot_cell_types_cluster(self, figsize, path):

        _, all_celltypes = Celltype_Analyzer.default_celltypes()
        clusters_cta = self.cluster_cta

        clusters_cta.set_known_types(all_celltypes)
        celltype_colors = [x.get_color() for x in clusters_cta.get_known_types()]
        memberships = clusters_cta.memberships()
        memberships = memberships.iloc[[0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16], :] # switching order so unknown is not above outputs and RGNs before pre-outputs
        celltype_colors = [celltype_colors[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,17,13,14,15,16]] # switching order so unknown is not above outputs and RGNs before pre-outputs

        ind = np.arange(0, len(clusters_cta.Celltypes))

        plt.bar(ind, memberships.iloc[0, :], color=celltype_colors[0])
        bottom = memberships.iloc[0, :]
        for i in range(1, len(memberships.index)):
            plt.bar(ind, memberships.iloc[i, :], bottom = bottom, color=celltype_colors[i])
            bottom = bottom + memberships.iloc[i, :]
        plt.savefig(path, format='pdf', bbox_inches='tight')    