import networkx as nx
import numpy as np 
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import si_animator  
from sklearn.preprocessing import minmax_scale



#read data
fh = "aggregated_US_air_traffic_network_undir.edg"
G=nx.read_weighted_edgelist(fh, nodetype=int)
nodes_n = G.number_of_nodes()
nodes = list(G.nodes())

# print(G.edges)

ft = "events_US_air_traffic_GMT.txt"
f = open(ft,'r')
air_traffic_GMT = []

for line in f.readlines(): 
    if len(line) and (not line.startswith('S')):  
        i = line.split("\n")
        i = i[0].split(' ')
        air_traffic_GMT.append(i)
    else:
        continue
f.close()

#  sort all flights in increasing order of departure time
sorter = lambda x:(x[2])
air_traffic_GMT_sorted = sorted(air_traffic_GMT,key=sorter)


print("data read done")

#dict store
def SI_model(n,p, immunized=[]):
    
    infection_times={}
    infection_times[n] = float(air_traffic_GMT_sorted[0][2])

    for val in air_traffic_GMT_sorted:
        source_node = int(val[0])
        destination_node = int(val[1])
        arrival_t = float(val[3])
        departure_t = float(val[2])
        if source_node in infection_times:
        
            if destination_node in infection_times:
                if arrival_t < infection_times[destination_node] and departure_t>infection_times[source_node]:
                    infection_times[destination_node] = arrival_t

                 
            else:
                if destination_node not in immunized and np.random.rand() < p and departure_t>infection_times[source_node] :
                    infection_times[destination_node] = arrival_t
                    if source_node == 41:
                        print(destination_node)
                

    return infection_times

def SI_model_links(n,p, g_links, immunized=[]):
    
    infection_times={}
    infection_times[n] = float(air_traffic_GMT_sorted[0][2])

    for val in air_traffic_GMT_sorted:
        source_node = int(val[0])
        destination_node = int(val[1])
        arrival_t = float(val[3])
        departure_t = float(val[2])

        if source_node in infection_times:
        
            if destination_node in infection_times:
                if arrival_t < infection_times[destination_node] and departure_t>infection_times[source_node]:
                    infection_times[destination_node] = arrival_t

                #     if g_links.has_edge(source_node,destination_node):
                #         g_links[source_node][destination_node]['weight']+=1
                #     else:
                #         g_links.add_edge(source_node,destination_node,weight=1)
                # elif arrival_t >= infection_times[destination_node] and departure_t>infection_times[source_node]:
                #     if g_links.has_edge(source_node,destination_node):
                #         g_links[source_node][destination_node]['weight']+=1
                #     else:
                #         g_links.add_edge(source_node,destination_node,weight=1)

            else:
                if destination_node not in immunized and np.random.rand() < p and departure_t>infection_times[source_node] :
                    infection_times[destination_node] = arrival_t
                    g_links[source_node][destination_node]['weight']+=1
                    # if g_links.has_edge(source_node,destination_node):
                    #     g_links[source_node][destination_node]['weight']+=1
                    # else:
                    #     g_links.add_edge(source_node,destination_node,weight=1)
                        #print("hello", source_node, destination_node)
                    # if source_node == 41:
                    #     print(destination_node)
                        
                
    return g_links


# prevalence 
def prevalence(n ,p, iter = 10, multi_n = False, immunized = [], n_multi=[]):
    #infection_times_multi = {}
    prevalence_multi = []

    if multi_n:
        N = n_multi
    else:
        N = [n]*iter
    i = 0
    while i < iter:
        infection_times_multi = SI_model(N[i],p, immunized)
        a = sorted(infection_times_multi.values())
        # print(len(a))
        num_steps = []
        for time in timestamps:
            num_steps.append(float(sum(a<=time))/nodes_n)
        prevalence_multi.append(num_steps) 
        i += 1
    return prevalence_multi


#-------------------- Task 1 --------------------#
def task1():
    print("start infecting")
    infection_times = SI_model(0,1)
    # infection_times_a = SI_model_a(0,1)
    print("infecting finished")

    print(infection_times[41])
    #print(infection_times_a[41])


#-------------------- Task 2 --------------------#

def task2():
    print("start infecting")
    P = [0.01, 0.05, 0.1, 0.5, 1.0]
    #P=[0.01]

    prevalence_multi_avg = []

    for p in P:
        prevalence_multi = prevalence(0, p)
        prevalence_multi_avg.append([float(sum(l))/len(l) for l in zip(*prevalence_multi)])

    #print(prevalence_multi_avg)

    print("infecting finished")

    # Task2 Visualization
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlabel('TimeStamps')
    ax.set_ylabel('Prevalence')
    ax.set_title("Effect of infection probability p on spreading speed")
    n = 0
    for prevalences in prevalence_multi_avg:
        plt.plot(list(timestamps), prevalences, label=P[n])
        n += 1
        ax.legend()

    plt.show()


#-------------------- Task 3 --------------------#
def task3():
    print("start infecting")

    N = [0, 4, 41, 100, 200]
    P = [0.1]

    prevalence_multi_avg = []

    for n in N:
        for p in P:
            prevalence_multi = prevalence(n, p)
            prevalence_multi_avg.append([float(sum(l))/len(l) for l in zip(*prevalence_multi)])

    #print(prevalence_multi_avg)

    print("infecting finished")

    # Task2 Visualization
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlabel('TimeStamps')
    ax.set_ylabel('Prevalence')
    ax.set_title("Effect of seed node selection on spreading speed")
    n = 0
    for prevalences in prevalence_multi_avg:
        plt.plot(list(timestamps), prevalences, label=N[n])
        n += 1
        ax.legend()

    plt.show()

#-------------------- Task 4 --------------------#
nodal_measures = {
    'unweighted clustering coefficient c': nx.clustering(G),
    'degree k': dict(G.degree()),
    'strength s': dict(G.degree(weight='weight')),
    'unweighted betweenness centrality': nx.betweenness_centrality(G),
    }

def task4():
    print("start infecting")

    p = 0.5
    iter = 50
    node_infected_t = {}
    median = {}
    for i in range(iter):
        n = int(np.random.randint(0,279))
        infection_times = SI_model(n,p)
        for key, value in infection_times.items():
            if key not in node_infected_t:
                node_infected_t[key] = [value]
            else:
                node_infected_t[key].append(value)
    for key, value in node_infected_t.items():
        median[key] = np.median(value)
        
    print(median[0])
    print("infecting finished")

    #Visualization

    spearmanr_ls = []

    for i, (title, info) in enumerate(nodal_measures.items()):
        x = [info[node] for node in nodes]
        y = [median[int(node)] for node in nodes]
        rho = sp.stats.spearmanr(x, y).correlation
        spearmanr_ls.append((title, rho))
    
        # normalized
        #x = minmax_scale(x)
        x, y = minmax_scale(x), minmax_scale(y)
        plt.figure(figsize=(10,6))
        plt.xlabel(f'{title} (Normalized)')
        plt.ylabel('Median Infection time (Normalized)')
        plt.scatter(x, y)
        plt.title(f'Infect time vs {title} (normalized)\n Spearman r:{rho}')
    
        plt.savefig(f'result/{title}')
        #plt.show()
    
    for measure, rho in spearmanr_ls:
        print(f'{measure}: {rho}')


#-------------------- Task 5 --------------------#
def random_neighbour(n):
    immunized = []
    i=0
    node = np.random.choice(nodes, n)
    while i < n:
        node_neighbour = np.random.choice(list(G.neighbors(node[i])))
        immunized.append(int(node_neighbour))
        i += 1

    return immunized

def task5():
    iter = 20
    p = 0.5
    multi_n = True 
    immunized_n = 10

    #immunized methods and their immunized nodes
    immunized_methods = {}
    immunized_methods["random_neighbour"] = random_neighbour(immunized_n)
    immunized_methods["random"] = list(map(int, np.random.choice(nodes, immunized_n)))

    for key, value in nodal_measures.items():
        highest_nodes = [int(k) for k, v in sorted(value.items(), key=lambda item: item[1], reverse=True)]
        immunized_methods[key] = highest_nodes[:immunized_n]

    
    # seed nodes group
    immunied_nodes_g = []
    for key, value in immunized_methods.items():
        immunied_nodes_g = list(set(immunied_nodes_g).union(set(value)))
    unimmunied_nodes = list(set(nodes).difference(set(immunied_nodes_g)))
    n_multi = list(map(int, np.random.choice(unimmunied_nodes, iter)))

    #prevalence calculation
    prevalence_multi_avg = []
    for key, immunized in immunized_methods.items():
        print(immunized)
        prevalence_multi = prevalence(0, p, iter, multi_n, immunized, n_multi)
        prevalence_multi_avg.append([float(sum(l))/len(l) for l in zip(*prevalence_multi)])

    # Task5 Visualization
    keys = list(immunized_methods.keys())
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlabel('TimeStamps')
    ax.set_ylabel('Prevalence')
    ax.set_title("Effectiveness of Immunization strategy")
    n = 0
    for prevalences in prevalence_multi_avg:
        plt.plot(list(timestamps), prevalences, label=keys[n])
        n += 1
        ax.legend()

    plt.show()


#-------------------- Task 6 --------------------#
#link graph
def task6():
    iter = 20
    p = 0.5

    seed_nodes = list(map(int, np.random.choice(nodes, iter)))

    # transmission_count = {edge: 0 for edge in G.edges}
    g_links = nx.Graph()
    node = range(0,279)
    g_links.add_nodes_from(node)
    edges_ = list(G.edges())
    g_links.add_edges_from(edges_,weight=0)
    for n in seed_nodes:
        g_links = SI_model_links(n, p, g_links)
        print("kkkkk")


    weights = nx.get_edge_attributes(g_links,'weight')
    edges = list(g_links.edges())

    print(len(weights),"weights", len(edges), "edges")



    # Visualization
    airport = pd.read_csv('US_airport_id_info.csv')
    airport.head()
    xycoords = {airport.id[index]:
        (airport.xcoordviz[index],airport.ycoordviz[index])
        for index in airport.index}
    widths = [v/(iter) for v in weights.values()]
    # widths = [n/max(widths) for n in widths]
    # print(widths)

    si_animator.plot_network_usa(g_links, xycoords, edges=edges, linewidths=widths)
    plt.title("Transmission Links")
    plt.savefig('result/plot_network_usa.png')
    # plt.show()
    mst = nx.maximum_spanning_tree(g_links)
    mst_edges = list(mst.edges)
    si_animator.plot_network_usa(mst, xycoords, 
                 mst_edges, [1 for _ in mst_edges])
    plt.savefig('result/plot_mst.png')

    # scatter plots showing the fraction of times as a function of the following link properties

    scatter_info = {
        'weight': nx.get_edge_attributes(G, 'weight'),
        'unweighted link betweenness centrality': nx.edge_betweenness_centrality(g_links)
    }

    spearmanr_ls_links = []
    for attribute, info in scatter_info.items():   
        if attribute == 'weight':
            x = minmax_scale([info[edge] for edge in edges_])
        else:
            x = minmax_scale([info[edge] for edge in edges])

        # x = [info[edge] for edge in edges_]

        y = widths
        print(len(y),':y ', len(x),':x')
        rho = sp.spearmanr(x, y).correlation
        spearmanr_ls_links.append((attribute, rho))
    
        plt.figure(figsize=(8,5))
        plt.xlabel(f'{attribute} (normalized)')
        plt.ylabel('Transmission fraction $f_{ij}$')
        plt.scatter(x, y,alpha=0.9, s=30)
        plt.title(f'Transmission fraction vs {attribute}')

        plt.savefig(f'result/{attribute}')
        plt.show()
    
    print(spearmanr_ls_links)

#-------------- main ------------#

# timestamps
min_t = float(air_traffic_GMT_sorted[0][2])
max_t = float(sorted(air_traffic_GMT, key = lambda x:(x[3]), reverse = True)[0][3])

steps = 20
timestamps = np.linspace(min_t, max_t, steps)

# main 
#task1()
#task2()
#task3()
#task4()
#task5()
task6()
