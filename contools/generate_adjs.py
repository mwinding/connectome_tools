# %%
# significant portions below are from Ben Pedigo's script found here: https://github.com/neurodata/maggot_models/blob/052af5d5999b2ae689b9d4a8d398748858acfd3a/data/process_scripts/process_maggot_brain_connectome_2021-05-24.py
# parts of this script have been modified to make it more general

import json
import os
import pprint
import sys
import time
from datetime import date
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from pickle import dump

import matplotlib.pyplot as plt
import navis
import networkx as nx
import numpy as np
import pandas as pd
import pymaid
from requests.exceptions import ChunkedEncodingError
from contools.process_matrix import Promat
from contools.process_matrix import Adjacency_matrix

##########
# functions

def get_connectors(nl):
    connectors = pymaid.get_connectors(nl)
    connectors.set_index("connector_id", inplace=True)
    connectors.drop(
        [
            "confidence",
            "creation_time",
            "edition_time",
            "tags",
            "creator",
            "editor",
            "type",
        ],
        inplace=True,
        axis=1,
    )
    details = pymaid.get_connector_details(connectors.index.values)
    details.set_index("connector_id", inplace=True)
    connectors = pd.concat((connectors, details), ignore_index=False, axis=1)
    connectors.reset_index(inplace=True)
    return connectors


def _append_labeled_nodes(add_list, nodes, name):
    for node in nodes:
        add_list.append({"node_id": node, "node_type": name})


def _standard_split(n, treenode_info, splits):
    skid = int(n.skeleton_id)
    split_node = splits[skid]
    # order of output is axon, dendrite
    fragments = navis.cut_skeleton(n, split_node)

    # axon(s)
    for f in fragments[:-1]:
        axon_treenodes = f.nodes.node_id.values
        _append_labeled_nodes(treenode_info, axon_treenodes, "axon")

    # dendrite
    dendrite = fragments[-1]
    tags = dendrite.tags
    if "mw periphery" not in tags and "soma" not in tags:
        msg = f"WARNING: when splitting neuron {skid} ({n.name}), no soma or periphery tag was found on dendrite fragment."
        raise UserWarning(msg)
        print("Whole neuron tags:")
        pprint.pprint(n.tags)
        print("Axon fragment tags:")
        for f in fragments[:-1]:
            pprint.pprint(f.tags)
    dend_treenodes = fragments[-1].nodes.node_id.values
    _append_labeled_nodes(treenode_info, dend_treenodes, "dendrite")


def _special_mbon_split(n, treenode_info, output_path):
    skid = int(n.skeleton_id)
    axon_starts = list(
        pymaid.find_nodes(tags="mw axon start", skeleton_ids=skid)["node_id"]
    )
    axon_ends = list(
        pymaid.find_nodes(tags="mw axon end", skeleton_ids=skid)["node_id"]
    )
    axon_splits = axon_starts + axon_ends

    fragments = navis.cut_skeleton(n, axon_splits)
    axons = []
    dendrites = []
    for fragment in fragments:
        root = fragment.root
        if "mw axon start" in fragment.tags and root in fragment.tags["mw axon start"]:
            axons.append(fragment)
        elif "mw axon end" in fragment.tags and root in fragment.tags["mw axon end"]:
            dendrites.append(fragment)
        elif "soma" in fragment.tags and root in fragment.tags["soma"]:
            dendrites.append(fragment)
        else:
            raise UserWarning(
                f"Something weird happened when splitting special neuron {skid}"
            )

    for a in axons:
        axon_treenodes = a.nodes.node_id.values
        _append_labeled_nodes(treenode_info, axon_treenodes, "axon")

    for d in dendrites:
        dendrite_treenodes = d.nodes.node_id.values
        _append_labeled_nodes(treenode_info, dendrite_treenodes, "dendrite")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    navis.plot2d(axons, color="red", ax=ax)
    navis.plot2d(dendrites, color="blue", ax=ax)
    plt.savefig(
        output_path / f"weird-mbon-{skid}.png",
        facecolor="w",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.3,
        dpi=300,
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    navis.plot2d(axons, color="red", ax=ax)
    navis.plot2d(dendrites, color="blue", ax=ax)
    ax.azim = -90
    ax.elev = 0
    plt.savefig(
        output_path / f"weird-mbon-{skid}-top.png",
        facecolor="w",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.3,
        dpi=300,
    )


def get_treenode_types(nl, splits, special_ids, output_path):
    treenode_info = []
    print("Cutting neurons...")
    for i, n in enumerate(nl):
        skid = int(n.skeleton_id)

        if skid in special_ids:
            _special_mbon_split(n, treenode_info, output_path)
        elif skid in splits.index:
            _standard_split(n, treenode_info, splits)
        else:  # unsplittable neuron
            # TODO explicitly check that these are unsplittable
            unsplit_treenodes = n.nodes.node_id.values
            _append_labeled_nodes(treenode_info, unsplit_treenodes, "unsplit")

    treenode_df = pd.DataFrame(treenode_info)
    # a split node is included in pre and post synaptic fragments
    # here i am just removing, i hope there is never a synapse on that node...
    # NOTE: would probably throw an error later if there was
    treenode_df = treenode_df[~treenode_df["node_id"].duplicated(keep=False)]
    treenode_series = treenode_df.set_index("node_id")["node_type"]
    return treenode_series


def flatten_compartment_types(synaptic_type):
    if synaptic_type == "axon":
        return "a"
    elif synaptic_type == "dendrite" or synaptic_type == "unsplit":
        return "d"
    else:
        return "-"


def flatten(series):
    f = np.vectorize(flatten_compartment_types)
    arr = f(series)
    new_series = pd.Series(data=arr, index=series.index)
    return new_series


def connectors_to_nx_multi(connectors, meta_data_dict):
    g = nx.from_pandas_edgelist(
        connectors,
        source="presynaptic_to",
        target="postsynaptic_to",
        edge_attr=True,
        create_using=nx.MultiDiGraph,
    )
    nx.set_node_attributes(g, meta_data_dict)
    return g


def flatten_muligraph(multigraph, meta_data_dict):
    # REF: https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-into-simple-graph-with-weighted-edges
    g = nx.DiGraph()
    for node in multigraph.nodes():
        g.add_node(node)
    for i, j, data in multigraph.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if g.has_edge(i, j):
            g[i][j]["weight"] += w
        else:
            g.add_edge(i, j, weight=w)
    nx.set_node_attributes(g, meta_data_dict)
    return g


def adj_split_axons_dendrites(all_neurons, split_tag, special_split_tags, not_split_skids):
    # user must login to CATMAID instance before starting

    t0 = time.time()

    # find today's date and make an output folder with that name
    today = date.today()
    today = date.strftime(today, '%Y-%m-%d')

    output_path = Path(f"data/processed/{today}")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    print("Pulling neurons...\n")

    ids = all_neurons

    batch_size = 20
    max_tries = 10
    n_batches = int(np.floor(len(ids) / batch_size))
    if len(ids) % n_batches > 0:
        n_batches += 1
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {n_batches}")
    print(f"Number of neurons: {len(ids)}")
    print(f"Batch product: {n_batches * batch_size}\n")

    i = 0
    currtime = time.time()
    nl = pymaid.get_neuron(
        ids[i * batch_size : (i + 1) * batch_size], with_connectors=False
    )
    print(f"{time.time() - currtime:.3f} seconds elapsed for batch {i}.")
    for i in range(1, n_batches):
        currtime = time.time()
        n_tries = 0
        success = False
        while not success and n_tries < max_tries:
            try:
                nl += pymaid.get_neuron(
                    ids[i * batch_size : (i + 1) * batch_size], with_connectors=False
                )
                success = True
            except ChunkedEncodingError:
                print(f"Failed pull on batch {i}, trying again...")
                n_tries += 1
        print(f"{time.time() - currtime:.3f} seconds elapsed for batch {i}.")

    print("\nPulled all neurons.\b")


    print("\nPickling neurons...")
    currtime = time.time()

    with open(output_path / "neurons.pickle", "wb") as f:
        dump(nl, f)
    print(f"{time.time() - currtime:.3f} seconds elapsed to pickle.")


    print("Pulling split points and special split neuron ids...")
    currtime = time.time()

    ##########
    # double-check for issues with split tags

    # find neurons and nodes with split tag
    splits = pymaid.find_nodes(tags=split_tag)
    splits = splits.set_index("skeleton_id")["node_id"].squeeze()

    # find neurons and nodes with special split start tag
    special_splits_start = pymaid.find_nodes(tags=special_split_tags[0])
    special_splits_start = special_splits_start.set_index("skeleton_id")["node_id"].squeeze()

    # find neurons and nodes with special split end tag
    special_splits_end = pymaid.find_nodes(tags=special_split_tags[1])
    special_splits_end = special_splits_end.set_index("skeleton_id")["node_id"].squeeze()

    # every skeleton with special split start should also have a special split end
    special_start_ids = np.unique(special_splits_start.index)
    special_end_ids = np.unique(special_splits_end.index)
    intersect = np.intersect1d(special_start_ids, special_end_ids)
    union = np.union1d(special_start_ids, special_end_ids)
    if(len(intersect)!=len(union)):
        print('Not all neurons with complex splits have the proper tags!')
        sys.exit(f'Check tags in the following skids: {list(np.setdiff1d(union, intersect))}')


    # split tag skeletons and special split tag skeletons should be mutually exclusive 
    split_ids = list(splits.index)
    special_ids = list(union)

    if(len(np.intersect1d(split_ids, special_ids))!=0):
        sys.exit(f'Splitting error! Check {problem_skids} which have a combination of {split_tag}, {special_split_tags[0]}, and {special_split_tags[1]} tags')


    print(f"{time.time() - currtime:.3f} elapsed.\n")

    # all splittable neurons should have split tags
    should_not_split = not_split_skids # unsplittable neurons defined by user
    should_split = list(np.setdiff1d(all_neurons, should_not_split))

    not_split = list(np.setdiff1d(should_split, split_ids + special_ids))

    if len(not_split) > 0:
        print(
            f"WARNING: {len(not_split)} neurons should have had split tag(s) and didn't:"
        )
        print(not_split)
        sys.exit('Check the above skeleton IDs')

    ########
    # processing data

    print("Getting treenode compartment types...")
    currtime = time.time()
    treenode_types = get_treenode_types(nl, splits, special_ids, output_path)
    print(f"{time.time() - currtime:.3f} elapsed.\n")


    print("Pulling connectors...\n")
    currtime = time.time()
    connectors = get_connectors(nl)
    print(f"{time.time() - currtime:.3f} elapsed.\n")

    explode_cols = ["postsynaptic_to", "postsynaptic_to_node"]
    index_cols = np.setdiff1d(connectors.columns, explode_cols)

    print("Exploding connector DataFrame...")
    # explode the lists within the connectors dataframe
    connectors = (
        connectors.set_index(list(index_cols)).apply(pd.Series.explode).reset_index()
    )
    # TODO figure out these nans
    bad_connectors = connectors[connectors.isnull().any(axis=1)]
    bad_connectors.to_csv(output_path / "bad_connectors.csv")
    # connectors = connectors[~connectors.isnull().any(axis=1)]

    print(f"Connectors with errors: {len(bad_connectors)}")
    connectors = connectors.astype(
        {
            "presynaptic_to": "Int64",
            "presynaptic_to_node": "Int64",
            "postsynaptic_to": "Int64",
            "postsynaptic_to_node": "Int64",
        }
    )

    print("Applying treenode types to connectors...")
    currtime = time.time()
    connectors["presynaptic_type"] = connectors["presynaptic_to_node"].map(treenode_types)
    connectors["postsynaptic_type"] = connectors["postsynaptic_to_node"].map(treenode_types)

    connectors["in_subgraph"] = connectors["presynaptic_to"].isin(ids) & connectors[
        "postsynaptic_to"
    ].isin(ids)
    print(f"{time.time() - currtime:.3f} elapsed.\n")


    meta = pd.DataFrame(index=all_neurons)

    print("Calculating neuron total inputs and outputs...")
    axon_output_map = (
        connectors[connectors["presynaptic_type"] == "axon"]
        .groupby("presynaptic_to")
        .size()
    )
    axon_input_map = (
        connectors[connectors["postsynaptic_type"] == "axon"]
        .groupby("postsynaptic_to")
        .size()
    )

    dendrite_output_map = (
        connectors[connectors["presynaptic_type"].isin(["dendrite", "unsplit"])]
        .groupby("presynaptic_to")
        .size()
    )
    dendrite_input_map = (
        connectors[connectors["postsynaptic_type"].isin(["dendrite", "unsplit"])]
        .groupby("postsynaptic_to")
        .size()
    )
    meta["axon_output"] = meta.index.map(axon_output_map).fillna(0.0)
    meta["axon_input"] = meta.index.map(axon_input_map).fillna(0.0)
    meta["dendrite_output"] = meta.index.map(dendrite_output_map).fillna(0.0)
    meta["dendrite_input"] = meta.index.map(dendrite_input_map).fillna(0.0)
    print()

    # remap the true compartment type mappings to the 4 that we usually use


    connectors["compartment_type"] = flatten(connectors["presynaptic_type"]) + flatten(
        connectors["postsynaptic_type"]
    )


    subgraph_connectors = connectors[connectors["in_subgraph"]]
    meta_data_dict = meta.to_dict(orient="index")


    full_g = connectors_to_nx_multi(subgraph_connectors, meta_data_dict)

    graph_types = ["aa", "ad", "da", "dd"]
    color_multigraphs = {}
    color_flat_graphs = {}
    for graph_type in graph_types:
        color_subgraph_connectors = subgraph_connectors[
            subgraph_connectors["compartment_type"] == graph_type
        ]
        color_g = connectors_to_nx_multi(color_subgraph_connectors, meta_data_dict)
        color_multigraphs[graph_type] = color_g
        flat_color_g = flatten_muligraph(color_g, meta_data_dict)
        color_flat_graphs[graph_type] = flat_color_g

    flat_g = flatten_muligraph(full_g, meta_data_dict)

    ########
    # saving data
    print("Saving metadata as csv...")
    meta.to_csv(output_path / "meta_data.csv")
    meta.to_csv(output_path / "meta_data_unmodified.csv")

    print("Saving connectors as csv...")
    connectors.to_csv(output_path / "connectors.csv")

    print("Saving each flattened color graph as graphml...")
    for graph_type in graph_types:
        nx.write_graphml(
            color_flat_graphs[graph_type], output_path / f"G{graph_type}.graphml"
        )
    nx.write_graphml(flat_g, output_path / "G.graphml")


    print("Saving each flattened color graph as txt edgelist...")
    for graph_type in graph_types:
        nx.write_weighted_edgelist(
            color_flat_graphs[graph_type], output_path / f"G{graph_type}_edgelist.txt"
        )
    nx.write_weighted_edgelist(flat_g, output_path / "G_edgelist.txt")

    ##########
    # saving data as pandas DataFrame adjacencies

    # import graphs of axon-dendrite split data; generated by Ben Pedigo's scripts
    G = nx.readwrite.graphml.read_graphml(f'data/processed/{today}/G.graphml', node_type=int)
    Gad = nx.readwrite.graphml.read_graphml(f'data/processed/{today}/Gad.graphml', node_type=int)
    Gaa = nx.readwrite.graphml.read_graphml(f'data/processed/{today}/Gaa.graphml', node_type=int)
    Gdd = nx.readwrite.graphml.read_graphml(f'data/processed/{today}/Gdd.graphml', node_type=int)
    Gda = nx.readwrite.graphml.read_graphml(f'data/processed/{today}/Gda.graphml', node_type=int)

    print("Generating split adjacency matrices as csv...")

    # generate adjacency matrices
    adj_all = pd.DataFrame(nx.adjacency_matrix(G=G, weight = 'weight').todense(), columns = G.nodes, index = G.nodes)
    adj_ad = pd.DataFrame(nx.adjacency_matrix(G=Gad, weight = 'weight').todense(), columns = Gad.nodes, index = Gad.nodes)
    adj_aa = pd.DataFrame(nx.adjacency_matrix(G=Gaa, weight = 'weight').todense(), columns = Gaa.nodes, index = Gaa.nodes)
    adj_dd = pd.DataFrame(nx.adjacency_matrix(G=Gdd, weight = 'weight').todense(), columns = Gdd.nodes, index = Gdd.nodes)
    adj_da = pd.DataFrame(nx.adjacency_matrix(G=Gda, weight = 'weight').todense(), columns = Gda.nodes, index = Gda.nodes)

    # add back in skids with no edges (with rows/columns of 0); simplifies some later analysis
    def refill_adjs(adj, adj_all):
        skids_diff = np.setdiff1d(adj_all.index, adj.index)
        adj = adj.append(pd.DataFrame([[0]*adj.shape[1]]*len(skids_diff), index = skids_diff, columns = adj.columns)) # add in rows with 0s
        for skid in skids_diff:
            adj[skid]=[0]*len(adj.index)
        return(adj)

    adj_ad = refill_adjs(adj_ad, adj_all)
    adj_aa = refill_adjs(adj_aa, adj_all)
    adj_dd = refill_adjs(adj_dd, adj_all)
    adj_da = refill_adjs(adj_da, adj_all)

    #convert column names to int for easier indexing
    adj_all.columns = adj_all.columns.astype(int)
    adj_ad.columns = adj_ad.columns.astype(int)
    adj_aa.columns = adj_aa.columns.astype(int)
    adj_da.columns = adj_da.columns.astype(int)
    adj_dd.columns = adj_dd.columns.astype(int)

    # export adjacency matrices
    adj_all.to_csv(f'data/adj/all-all_{today}.csv')
    adj_ad.to_csv(f'data/adj/ad_{today}.csv')
    adj_aa.to_csv(f'data/adj/aa_{today}.csv')
    adj_dd.to_csv(f'data/adj/dd_{today}.csv')
    adj_da.to_csv(f'data/adj/da_{today}.csv')

    # import input data and export as simplified csv
    meta_data = pd.read_csv(f'data/processed/{today}/meta_data.csv', index_col = 0)
    inputs = meta_data.loc[:, ['axon_input', 'dendrite_input']]
    outputs = meta_data.loc[:, ['axon_output', 'dendrite_output']]

    # exporting input data
    inputs.to_csv(f'data/adj/inputs_{today}.csv')
    outputs.to_csv(f'data/adj/outputs_{today}.csv')

    print()
    print()
    print("Done!")

    elapsed = time.time() - t0
    delta = timedelta(seconds=elapsed)
    print("----")
    print(f"{delta} elapsed for whole script.")
    print("----")

    sys.stdout.close()

def edge_thresholds(path, threshold, left_annot, right_annot, pairs, date = date.strftime(date.today(), '%Y-%m-%d') ): # default assumes data is from today
# threshold units are in %input on dendrite or axon

    adj_all = pd.read_csv(f'{path}/all-all_{date}.csv', index_col = 0).rename(columns=int)
    adj_ad = pd.read_csv(f'{path}/ad_{date}.csv', index_col = 0).rename(columns=int)
    adj_aa = pd.read_csv(f'{path}/aa_{date}.csv', index_col = 0).rename(columns=int)
    adj_dd = pd.read_csv(f'{path}/dd_{date}.csv', index_col = 0).rename(columns=int)
    adj_da = pd.read_csv(f'{path}/da_{date}.csv', index_col = 0).rename(columns=int)

    inputs = pd.read_csv(f'{path}/inputs_{date}.csv', index_col = 0)

    # initialize adj matrices
    adj_all_mat = Adjacency_matrix(adj_all, inputs, 'summed', pairs=pairs)
    adj_ad_mat = Adjacency_matrix(adj_ad, inputs, 'ad', pairs=pairs)
    adj_aa_mat = Adjacency_matrix(adj_aa, inputs, 'aa', pairs=pairs)
    adj_dd_mat = Adjacency_matrix(adj_dd, inputs, 'dd', pairs=pairs)
    adj_da_mat = Adjacency_matrix(adj_da, inputs, 'da', pairs=pairs)

    # generate all paired and nonpaired edges from each matrix with threshold
    # export as paired edges between paired neurons (collapse left/right hemispheres, except for nonpaired neurons)
    # export as normal edge list, but with pair-wise threshold

    adjs = [adj_all_mat, adj_ad_mat, adj_aa_mat, adj_dd_mat, adj_da_mat]
    adjs_names = ['summed', 'ad', 'aa', 'dd', 'da']

    left = Promat.get_hemis(left_annot, right_annot, side='left')
    right = Promat.get_hemis(left_annot, right_annot, side='right')

    for i, adj_mat in enumerate(adjs):
        matrix_pairs = Promat.extract_pairs_from_list(adj_mat.skids, pairs)
        matrix_nonpaired = list(np.intersect1d(matrix_pairs[2].nonpaired, left+right)) # ignore unipolar neurons, not in set of brain neurons
        all_sources = list(matrix_pairs[0].leftid) + matrix_nonpaired

        all_edges_combined = adj_mat.threshold_edge_list(all_sources, matrix_nonpaired, threshold, left, right) # currently generates edge list for all paired -> paired/nonpaired, nonpaired -> paired/nonpaired
        all_edges_combined.to_csv(f'data/edges_threshold/{adjs_names[i]}_all-paired-edges_{date}.csv')
        all_edges_split = adj_mat.split_paired_edges(all_edges_combined, left, right)
        all_edges_split.to_csv(f'data/edges_threshold/pairwise-threshold_{adjs_names[i]}_all-edges_{date}.csv')