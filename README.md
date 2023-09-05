connectome_tools
==============================
tools to analyze connectomics data from a CATMAID instance; includes modified scripts from neurodata/maggot_models by bdpedigo and from mwinding/connectome_analysis

Version 2.0.0: includes signal cascade algorithm that can interpret excitatory and inhibitory connections

Installation and setup
--------
connectome tools can be installed via pip:

```
pip install git+https://github.com/mwinding/connectome_tools
```

Classes can be imported as follows:

```
from contools import Prograph, Promat
from contools import Celltype, Celltype_Analyzer
from contools import Cascade_Analyzer
```

Project Organization
------------
```
├── LICENSE
├── README.md
├── contools                   <- contains 5 different analysis classes
│   │
│   ├── traverse/              <- guts of signal cascade algorithm and others
│   ├── cascade_analysis.py    <- Cascade_Analyzer(), a class facilitating analysis of signal cascade data
│   ├── celltype.py            <- Celltype(), a general celltype class containing skeleton IDs, plotting colors, and names
│   │                             Celltype_Analyzer(), which facilitates comparison and analysis of celltypes
│   ├── generate_adjs.py       <- contains functions needed to generate required datasets
│   ├── process_graph.py       <- Analyze_Nx_G(), enables new networkx functionality, such as double-edge swaps in directed graphs; 
│   │                             Prograph(), with many graph modification convenience methods
│   └── process_matrix.py      <- Adjacency_matrix(), allows pair-wise thresholding of adj_matrices; Promat(), many CATMAID convenience methods
│
└── examples
    ├── preparing-data_example.ipynb  <- example of how to generate data from CATMAID in a new project
    └── cascade_example.ipynb         <- signal cascade example, investigating polysynaptic information flow across a brain
                                      <- more examples coming, including multihop connectivity matrices, chromosome plots, marginal cluster plots, etc.

```
