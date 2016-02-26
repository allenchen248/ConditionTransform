import ast
import random
import string
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from collections import defaultdict

class NodeGrapher(ast.NodeVisitor):
	"""
	It didn't look like there was a good AST visualization tool online,
	so I wrote this class to cover that.
	"""
	def __init__(self, tree, ax=None):
		self.graph = defaultdict(lambda: [])
		self.labels = {}
		self.ax = ax

		super(NodeGrapher, self).__init__()

		try:
			if tree.__class__ == str:
				return self.visit(ast.parse(tree))
			else:
				return self.visit(tree)
		except TypeError as e:
				raise TypeError("Input Type Incorrect - either tree or string")

	def __str__(self):
		"""
		Lets us see into what the graph looks like (but while being lazy)
		"""
		return str(self.graph)

	def __repr__(self):
		"""
		Still being lazy
		"""
		return self.__str__()

	def make_label(self, node):
		"""
		In order to generate what the graph should look like, we need a way
		to determine what the unique nodes are.

		This function has a string representation of where each node is in memory,
		and uses this to determine where unique nodes are.
		"""
		curstring = str(node.__class__)[13:-2]
		if isinstance(node, ast.Name):
			curstring = node.id
		elif isinstance(node, ast.Num):
			curstring = str(node.n)
		elif isinstance(node, ast.Str):
			curstring = node.s

		if isinstance(node, ast.Load) or isinstance(node, ast.Store) or \
			isinstance(node, ast.Param) or isinstance(node, ast.Add) or \
			isinstance(node, ast.Sub) or isinstance(node, ast.Mult):
			return None

		try:
			self.labels[str(node)] = curstring
			return str(node)
		except AttributeError:
			return None

	def generic_visit(self, node):
		"""
		We use the visit method - it's called in every step of the visit
		function that we call (please don't overload that!), so we will see
		the entire graph
		"""
		for attr, vs in ast.iter_fields(node):
			if isinstance(vs, ast.AST):
				self.graph[self.make_label(node)].append(self.make_label(vs))
				self.visit(vs)
			elif isinstance(vs, list):
				for v in vs:
					if isinstance(v, ast.AST):
						self.graph[self.make_label(node)].append(self.make_label(v))
						self.visit(v)
			else:
				pass

	def draw_graph(self):
		"""
		Draws a networkx graph from the tree that is passed into
		the class.

		Generates positions using a DFS for pretty graphing.
		"""
		if None in self.graph:
			del self.graph[None]

		for vs in self.graph.itervalues():
			to_delete = []
			for i in xrange(len(vs)):
				if vs[i] is None:
					to_delete.append(i)

			for i in reversed(to_delete):
				del vs[i]

		self.G=nx.Graph(self.graph)

		for k,v in self.labels.iteritems():
			if v[:6] == 'Module':
				root = k
				break

		return self.__dfs_plot(root)

	def onclick(self, event):
		plt.close()

		root = self.__closest_point(event.xdata, event.ydata)

		dict_delete = []
		for k,vs in self.graph.iteritems():
			try:
				if (self.pos[k][1] >= event.ydata) and (k != root):
					dict_delete.append(k)
				else:
					to_delete = []
					for i in xrange(len(vs)):
						if self.pos[vs[i]][1] >= event.ydata:
							to_delete.append(i)

					for i in reversed(to_delete):
						print "Delete Edge: "+k+", "+vs[i]
						del vs[i]
			except KeyError:
				dict_delete.append(k)

		for k in dict_delete:
			print "DELETING: "+k
			del self.graph[k]
			del self.labels[k]

		self.G = nx.Graph(self.graph)
		return self.__dfs_plot(root)

	def __closest_point(self, xval, yval):
		try:
			allpos = self.pos.items()
			dists = [np.abs(x-xval)/10.+np.abs(y-yval) for _, (x, y) in allpos]
			return allpos[np.argmin(dists)][0]
		except AttributeError as e:
			raise AttributeError("Draw was not called first!")

	def __dfs_plot(self, root, graph=None, levels=None):
		levels = {k:None for k in self.labels.iterkeys()}

		levels[root] = (0,0,1)
		queue = [root]

		while len(queue) > 0:
			val = queue.pop()
			h, minval, maxval = levels[val]
			if len(self.graph[val]) > 0:
				diff = float(maxval-minval)/len(self.graph[val])
				for i,v in enumerate(self.graph[val]):
					try:
						if levels[v] is None:
							levels[v] = (h+1,minval+i*diff, minval+(i+1)*diff)
							queue.append(v)
					except KeyError:
						pass
					
		diff = 2./(1+len([1 for v in levels.itervalues() if v is None]))
		#self.pos = {k: (diff*(1+1),1) if v is None else (v[1]+v[2], -1*v[0]) for k,v in levels.iteritems()}
		self.pos = {k:(diff,1) if v is None else (v[1]+v[2], -1*v[0]) for k,v in levels.iteritems()}

		#if self.ax is None:
		self.fig = plt.figure()
		self.ax = plt.gca()
		self.fig.canvas.mpl_connect('button_press_event', lambda event:self.onclick(event))

		nx.draw(self.G, self.pos, node_color='w', node_shape='', \
			width=3., node_size = 600, alpha=0.1, ax=self.ax)
		nx.draw_networkx_labels(self.G, self.pos, \
			labels={k:self.labels[k] for k,v in levels.iteritems() if v is not None}, \
			ax=self.ax)

		plt.show(block=False)

		return self.fig