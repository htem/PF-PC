# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx


### load graphs
G_AB_con = nx.read_graphml("htem/PF_PC/connectivity_analysis/graphs/G_AB_con.graphml") 
G_AB_touch = nx.read_graphml("htem/PF_PC/connectivity_analysis/graphs/G_AB_con.graphml") 


### extract cell types
node_types = nx.get_node_attributes(G_AB_con, "type")

PC_set  = [n for n, t in node_types.items() if t == "pc"]
GPF_set  = [n for n, t in node_types.items() if t == "gpf"] #parallel fibers of local granule cells
AB_set = [n for n, t in node_types.items() if t == "ab"] #ascending branches of local granule cells
