# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx


### load graphs
G_con = nx.read_graphml("htem/PF_PC/connectivity_analysis/graphs/G_con.graphml") 
G_touch = nx.read_graphml("htem/PF_PC/connectivity_analysis/graphs/G_con.graphml") 


### extract cell types
node_types = nx.get_node_attributes(G_con, "type")

PC_set  = [n for n, t in node_types.items() if t == "pc"]
PF_set  = [n for n, t in node_types.items() if t == "remote_pf"]
GrC_set = [n for n, t in node_types.items() if t == "local_grc"]


### EXAMPLE: compute remote PF-PC connection probability distribution
G_con_sub = G_con.subgraph(PC_set + PF_set)
G_touch_sub = G_touch.subgraph(PC_set + PF_set)

con_probs = [
    G_con_sub.degree(pc) / G_touch_sub.degree(pc)
    for pc in PC_set
    ]

print("PF-PC connection probability: ", np.mean(con_probs))
