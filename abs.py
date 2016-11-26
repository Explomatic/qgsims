import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.linalg as linalg
import networkx as nx
import math
import time 
import sys

def ranf():
	return round((2/math.pi)*math.atan(random.expovariate(1.5)))


def random_adjacency_matrix(n):
    #matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    matrix = [[ranf() for i in range(n)] for j in range(n)]

    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0

    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]

    return np.matrix(matrix)

def add_edge(ram, node):
	N = ram.shape[0]
	if random.randint(0,1) == 1:
		temp = random.randint(0,N-1)
		ram[node,temp] = 1
		ram[temp,node] = 1
		ram[node,node] = 0
		print 'Edge added to node #' + str(node) + ' from node #' + str(temp)

	return ram

def remove_edge(ram, node):
	tempv = ram[:,node]
	save_idx = []
	for idx, val in enumerate(tempv):
		if val == 1:
			save_idx += [idx]
			
	try:
		temp = random.randint(0,len(save_idx)-1)
		if random.randint(0,1) == 1:
			ram[node,save_idx[temp]] = 0
			ram[save_idx[temp],node] = 0
			print 'Edge removed from node #' + str(node) + ' to node #' + str(save_idx[temp])
	except ValueError:
		print 'No edges to remove'
		return ram

	return ram

def update_conn(ram, psi):
	for idx, val in enumerate(psi):
		if abs(val) > 1/math.sqrt(N):
			ram = add_edge(ram, idx)

		else:
			ram = remove_edge(ram, idx)

	return ram

def update_graph(adjacency_matrix, grobj):
	rows, cols = np.where(adjacency_matrix == 1)
	edges = zip(rows.tolist(), cols.tolist())
	grobj.add_edges_from(edges)

	rows, cols = np.where(adjacency_matrix == 0)
	edges = zip(rows.tolist(), cols.tolist())
	grobj.remove_edges_from(edges)

	return grobj

def show_graph(adjacency_matrix):
	# given an adjacency matrix use networkx and matlpotlib to plot the graph
	n = 10
	rows, cols = np.where(adjacency_matrix == 1)
	edges = zip(rows.tolist(), cols.tolist())
	gr = nx.Graph()

	labels = {}
	for i in xrange(0,10):
		labels[i] = str(i)

	size = []
	for k in xrange(0,10):
		xcoord = math.cos(2*math.pi*k/n)
		ycoord = math.sin(2*math.pi*k/n)
		gr.add_node(k, pos=(xcoord,ycoord))
		size += [500]

	gr.add_edges_from(edges)

	pos=nx.get_node_attributes(gr,'pos')
	nx.draw(gr, pos, node_size=size)
	nx.draw_networkx_labels(gr, pos, labels, font_size=16)

	# nx.draw(gr) # edited to include labels

	#coords = nx.get_node_attributes(gr, 'pos')
	#print coords

	#nx.draw_networkx(gr)

	# now if you decide you don't want labels because your graph
	# is too busy just do: nx.draw_networkx(G,with_labels=False)
	plt.ion()
	#plt.show()
	return gr

N = 10 #Number of nodes in system
labels = {}
for i in xrange(0,10):
	labels[i] = str(i)

t = 1 #Hopping strength
ram = random_adjacency_matrix(N)
H = -t*ram

w, v = np.linalg.eig(H)
#psi_init = v[:,0]
psi_init = np.matrix('1;0;0;0;0;0;0;0;0;0')
#print v[:,0]

#sys.exit(1)

dt = 0.5 #The tiem for the rhyme
U = linalg.expm(-1j*H*dt)

tid = 0
graphobj = show_graph(ram)
pos = nx.get_node_attributes(graphobj, 'pos')

#time.sleep(1)
#
#plt.show()
psi = psi_init
print 'time is: 0'
while tid <= 5:
	density = np.absolute(psi)
	node_sizes = [(500)*(1+x) for x in density]
	time.sleep(0.5)
	plt.clf()
	#nx.draw_networkx_nodes(graphobj, pos, node_size=node_sizes)
	#nx.draw_networkx_edges(graphobj, pos, alpha=0.3)
	nx.draw(graphobj, pos, node_size=node_sizes)
	nx.draw_networkx_labels(graphobj, pos, labels, font_size=16)
	plt.show()
	plt.pause(0.05)
	tid += dt

	if tid % 3 == 0:
		raw_input('1111')
		print np.absolute(psi)*math.sqrt(N)
		ram = update_conn(ram, psi)
		H = -t*ram
		U = linalg.expm(-1j*H*dt)
		plt.clf()
		graphobj = update_graph(ram, graphobj)
		nx.draw(graphobj, pos, node_size=node_sizes)
		nx.draw_networkx_labels(graphobj, pos, labels, font_size=16)
		plt.show()
		plt.pause(0.05)		
		raw_input('2222')

	psi = np.dot(U, psi)
	print 'time is: ' + str(tid)
	

raw_input('sjdhfkjsd')









