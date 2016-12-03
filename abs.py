import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import scipy.linalg as linalg
import networkx as nx
import math
import time
import sys

def ranf():
	return(round((2/math.pi)*math.atan(random.expovariate(1.5))))


def init_state(N):
	psi = np.zeros(N)
	psi[0] = 1
	return(psi)


def node_labels(N):
	labels = {}
	for i in range(0,N):
		labels[i] = str(i)

	return(labels)


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

    return(np.matrix(matrix))


def add_edge(ram, nodek):
	conn_nodek = ram[:, nodek]

	conn_nodes = []
	for nodenum, conn in enumerate(conn_nodek):
		if conn == 0:
			conn_nodes += [nodenum]

	N = len(conn_nodes)
	try:
		rand_idx = random.randint(0,N-1)
		node_add = conn_nodes[rand_idx]
		ram[nodek, node_add] = 1
		ram[node_add, nodek] = 1
		ram[nodek, nodek] = 0
		print('Connection from node #{} to node #{} added.'.format(nodek, node_add))
		return(ram)
	except ValueError:
		print('No connections available.')
		return(ram)


def remove_edge(ram, nodek):
	conn_nodek = ram[:, nodek]
	
	conn_nodes = []
	for nodenum, conn in enumerate(conn_nodek):
		if conn == 1:
			conn_nodes += [nodenum]
	
	N_conns = len(conn_nodes)
	try:
		rand_idx = random.randint(0,N_conns-1)
		node_rm = conn_nodes[rand_idx]
		ram[nodek, node_rm] = 0
		ram[node_rm, nodek] = 0
		print('Connection removed from node #{} to node #{}'.format(nodek, node_rm))
		return(ram)
	except ValueError:
		print('No connections to remove.')
		return(ram)


def update_conn(ram, psi, a, b):
	N = ram.shape[0]
	for nodenum, density in enumerate(psi):
		pk = np.absolute(density)
		if pk < a:
			ram = remove_edge(ram, nodenum)
		elif pk > b:
			ram = add_edge(ram, nodenum)

	return(ram)


def update_graph(adjacency_matrix, grobj):
	rows, cols = np.where(adjacency_matrix == 1)
	edges = zip(rows.tolist(), cols.tolist())
	grobj.add_edges_from(edges)

	rows, cols = np.where(adjacency_matrix == 0)
	edges = zip(rows.tolist(), cols.tolist())
	grobj.remove_edges_from(edges)

	return(grobj)


def init_graph(adjacency_matrix, n):
	# given an adjacency matrix use networkx and matlpotlib to plot the graph
	rows, cols = np.where(adjacency_matrix == 1)
	edges = zip(rows.tolist(), cols.tolist())
	gr = nx.Graph()

	labels = {i:str(i) for x in range(n)}

	size = [500 for x in labels]
	for k in range(n):
		xcoord = (1/5)*n*math.cos(2*math.pi*k/n)
		ycoord = (1/5)*n*math.sin(2*math.pi*k/n)
		gr.add_node(k, pos=(xcoord,ycoord))

	gr.add_edges_from(edges)

	pos=nx.get_node_attributes(gr,'pos')
	fobj = plt.figure()
	nx.draw(gr, pos, node_size=size)
	nx.draw_networkx_labels(gr, pos, labels, font_size=16)

	# nx.draw(gr) # edited to include labels

	#coords = nx.get_node_attributes(gr, 'pos')
	#print coords

	#nx.draw_networkx(gr)

	# now if you decide you don't want labels because your graph
	# is too busy just do: nx.draw_networkx(G,with_labels=False)
	
	#plt.show()
	return(gr, fobj)


def program(N, T, ET):
	a = max(0, 0.2*(1-0.005*(N-10)))
	b = max(0, 0.35*(1-0.005*(N-10)))
	dt = 0.5 # time step
	currtime = 0 # current time
	time_limit = T # End time
	t = 1 # Hopping Strength
	edge_update = ET # Time interval between edge updating 
					# - i.e. either adding or removing edges

	plt.ion()

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
	graphobj, figobj = init_graph(ram, N)

	# Find the position of the nodes in the graph object
	pos = nx.get_node_attributes(graphobj, 'pos')

	print('Time is: 0')
	while currtime < time_limit:
		# Calculate probability density
		density = np.absolute(psi)
		
		# Rescale the nodes according to their density
		node_sizes = [(500)*(1+x) for x in density]

		# wait
		# time.sleep(0.3)

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
			ram = update_conn(ram, psi, a, b)

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

		# bool_density = density > b # Deprecated for the moment!
		if np.sum(ram) == 0:
			print('No more connections')
			break

		print('Time is: {}'.format(currtime))
		print('Total density: {}'.format(np.sum(np.absolute(psi))))
		

	input('sjdhfkjsd')


def main():
	args = sys.argv[1:]

	if not args:
		print('usage: nodes time edge_time')
		sys.exit(1)

	N = int(args[0])
	T = int(args[1])
	ET = int(args[2])

	program(N, T, ET)


if __name__ == '__main__':
  main()









