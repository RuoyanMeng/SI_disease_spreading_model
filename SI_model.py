import networkx as nx
import numpy as np 
import scipy.stats as sp
import matplotlib.pyplot as plt
import pandas as pd
import si_animator  
from sklearn.preprocessing import minmax_scale



#read data
fh = "aggregated_US_air_traffic_network_undir.edg"
G=nx.read_weighted_edgelist(fh)
nodes_n = G.number_of_nodes()

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
def SI_model(n,p):
    
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
                    #if destination_node == 41:
                        # print(infection_times[destination_node]," : update")
            else:
                if np.random.rand() < p and departure_t>infection_times[source_node]:
                    infection_times[destination_node] = arrival_t
                    #if destination_node == 41:
                        # print(infection_times[destination_node])

    return infection_times

# array store
def SI_model_a(n,p):
    infection_times= np.zeros(nodes_n)
    infection_times[n] = float(air_traffic_GMT_sorted[0][2])
    
    x = 0
    for val in air_traffic_GMT_sorted:
        source_node = int(val[0])
        destination_node = int(val[1])
        arrival_t = float(val[3])
        if infection_times[source_node] != 0:
            if infection_times[destination_node] !=0:
                if arrival_t < infection_times[destination_node]:
                    infection_times[destination_node] = arrival_t
                    #if destination_node == 41:
                        # print(infection_times[destination_node]," : update")
            else:
                if(np.random.rand() < p):
                    infection_times[destination_node] = arrival_t
                    #if destination_node == 41:
                        # print(infection_times[destination_node])

    return infection_times

# prevalence 
def prevalence(n,p, iter = 10):
    infection_times_multi = {}
    prevalence_multi = []

    i = 0
    while i < iter :
        infection_times_multi[i] = SI_model(n,p)
        a = sorted(infection_times_multi[i].values())
        print(len(a))
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

#-------------------- Task 3 --------------------#
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
    nodal_measures = {
        'unweighted clustering coefficient c': nx.clustering(G),
        'degree k': dict(G.degree()),
        'strength s': G.degree(weight='weight'),
        'unweighted betweenness centrality': nx.betweenness_centrality(G),
    }

    spearmanr_ls = []

    for i, (title, info) in enumerate(nodal_measures.items()):
        nodes = list(G.nodes())
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
        plt.show()
    
    for measure, rho in spearmanr_ls:
        print(f'{measure}: {rho}')
#Visualization
# si_animator.visualize_si(
#         infection_times_a,
#         save_fname="si_viz_example.gif",
#         tot_viz_time_in_seconds=10,
#         fps=10)



#-------------- main ------------#

# timestamps
min_t = float(air_traffic_GMT_sorted[0][2])
max_t = float(sorted(air_traffic_GMT, key = lambda x:(x[3]), reverse = True)[0][3])

steps = 20
timestamps = np.linspace(min_t, max_t, steps)

# main 
#task1()
#task2()
# task3()
task4()