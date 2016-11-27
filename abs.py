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

def init_state(N):
	psi = np.zeros(N)
	psi[0] = 1
	return psi

def node_labels(N):
	labels = {}
	for i in xrange(0,N):
		labels[i] = str(i)

	return labels

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
	N = ram.shape[0]
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

def show_graph(adjacency_matrix, n):
	# given an adjacency matrix use networkx and matlpotlib to plot the graph
	rows, cols = np.where(adjacency_matrix == 1)
	edges = zip(rows.tolist(), cols.tolist())
	gr = nx.Graph()

	labels = {}
	for i in xrange(0,n):
		labels[i] = str(i)

	size = []
	for k in xrange(0,n):
		xcoord = (1./5)*n*math.cos(2*math.pi*k/n)
		ycoord = (1./5)*n*math.sin(2*math.pi*k/n)
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

def program(N, T, ET):
	dt = 0.5 # time step
	currtime = 0 # current time
	time_limit = T # End time
	t = 1 # Hopping Strength
	edge_update = ET # Time interval between edge updating 
					# - i.e. either adding or removing edges

	# Create node labels
	labels = node_labels(N)

	# initialize random adjacency matrix
	ram = random_adjacency_matrix(N)

	# Create Hamiltonian for the system
	H = -t*ram

	# Find eigenvalues, w, and eigenvectors v of the Hamiltonian
	w, v = np.linalg.eig(H)

	# Initialize the state
	psi = init_state(N)

	# Calculate time-evolution operator
	U = linalg.expm(-1j*H*dt)

	# Create the graph object
	graphobj = show_graph(ram, N)

	# Find the position of the nodes in the graph object
	pos = nx.get_node_attributes(graphobj, 'pos')

	print 'time is: 0'
	while currtime < time_limit:
		# Calculate probability density
		density = np.absolute(psi)

		# Rescale the nodes according to their density
		node_sizes = [(500)*(1+x) for x in density]

		# wait
		time.sleep(0.5)

		# Clear the whole figure
		plt.clf()

		# Redraw figure
		nx.draw(graphobj, pos, node_size=node_sizes)

		# Redraw node labels
		nx.draw_networkx_labels(graphobj, pos, labels, font_size=16)

		# Show the graph
		plt.show()
		plt.pause(0.05)

		# increment the timestep
		currtime += dt

		# Add/remove edges
		if currtime % edge_update == 0:
			#raw_input('1111')
			#print np.absolute(psi)*math.sqrt(N)

			# Update the RAM
			ram = update_conn(ram, psi)

			# Recalculate the Hamiltonian and time-evolution operator
			H = -t*ram
			U = linalg.expm(-1j*H*dt)

			# Clear figure then redraw figure and node labels
			plt.clf()
			graphobj = update_graph(ram, graphobj)
			nx.draw(graphobj, pos, node_size=node_sizes)
			nx.draw_networkx_labels(graphobj, pos, labels, font_size=16)
			plt.show()
			plt.pause(0.05)


			#raw_input('2222')

		# Time-evovle state vector
		psi = np.dot(U, psi)
		print 'time is: ' + str(currtime)
		

	raw_input('sjdhfkjsd')

def main():
	args = sys.argv[1:]

	if not args:
		print 'usage: nodes time edge_time'
		sys.exit(1)

	N = int(args[0])
	T = int(args[1])
	ET = int(args[2])

	program(N, T, ET)

if __name__ == '__main__':
  main()









