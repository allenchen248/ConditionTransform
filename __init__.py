import ast
import random
import string
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from collections import defaultdict

stochastic_funcs = ['chimpRand', 'chimpNorm', 'Discrete']

sto_tree = """def test_func():
	a = .5
	b = a+12+chimpRand(10)
	c = b+a
	return c"""

chimp_tree = """class TestNormal(Dist):
    def __init__(self):
        self.discrete = Discrete([1,1],size=1)
    def run(self,trace):
        t = trace.chimplify("choice", self.discrete)
        return trace.chimplify("x",Normal(0,t*10,1))"""

class NameReplacer(ast.NodeTransformer):
	"""
	NameReplacer is a class that implements ast.NodeTransformer;
	it is used to walk along the nodes of an AST and performs computations
	on specific nodes as necessary (it calls the other steps recursively)
	"""
	def __init__(self, tree=None, print_trace=False, eval_funcs=False, stochastic=[], attrs=[], **kwargs):
		"""
		To replace variable names with constants, we can simulate trying to
		replace a variable "a" in a given function with the number 3. We would
		use the syntax:

		modified_tree = NameReplacer(a=3).visit(ast_tree)

		Right now only accepts numbers into kwargs replace. Eventually this will
		extend to non-numeric arguments. (the framework is in place)

		@param print_trace: Whether or not to print diagnostic information about
			what functions the class is working in

		@param eval_funcs: Whether or not to evaluate the removed functions during
			tree transformation. Be careful when using this - oftentimes, functions 
			won't exist in this namespace.

		@param attrs: A list of tuples ('self.discrete', 5) with the attribute name and the
			replacement value.

		@param stochastic: A list of functions that the user declares to be stochastic
			(so we can't replace them at compile time with a specific other program)

		@param **kwargs: Pass in any variable into the class for the variable's value
			to be statically assigned
		"""
		self.replace = kwargs
		self.eval_funcs = eval_funcs

		self.stochastic = {s:True for s in (stochastic+stochastic_funcs)}

		self.already_done = {k:False for k in kwargs.iterkeys()}

		self.discretevars = {}

		for name, val in attrs:
			self.already_done[name] = False
			self.replace[name] = val

		self.non_numeric = {}
		self.print_trace = print_trace
		self.stored_tree = None

		self.new_kwargs = []

		# TODO: Learn to flag instant return

		super(NameReplacer, self).__init__()

		if tree is not None:
			self.process(tree)

	@classmethod
	def from_file(cls, fname, func, read_mode='r', started=False, **kwargs):
		"""
		Classmethod that generates an instance of our function
		from a file and a function name.

		@param fname: the file name that we are reading from
		
		@param func: the function name (a string) within the file that
			we will be taking

		@param read_mode: the mode with which we are reading from the function
			(for use with the open command)

		@param started: Whether or not we should just start reading fromt he first line
			(that is, the file itself is meant to be run at command line)

		@param kwargs: Arguments that are passed into the class when we instantiate it
		"""
		func_text = ''
		indent = -1

		with open(fname, read_mode) as f:
			for line in f.readlines():
				pieces = line.lstrip().split(" ")

				if (len(line)-len(line.lstrip())) <= indent:
					break

				if (pieces[0] == 'def') and (pieces[1].split("(")[0] == func):
					indent = len(line)-len(line.lstrip())
					started = True

				if started:
					func_text += line[indent:]

		new_transformer = NameReplacer(**kwargs)
		new_transformer.stored_tree = [func, fname, func_text, None]
		return new_transformer

	def plot_compare(self):
		"""
		Compares two plots of a function - one from before the function was
		transformed and one from after.

		Doesn't quite work yet; plot.plot_processed=False doesn't work because
		the AST actually modifies the tree in place.
		"""
		self.plot(plot_processed=False)
		self.plot(plot_processed=True)

	def plot(self, data=None, plot_processed=True):
		"""
		Plots the AST prettily!

		@param data: The data that we want to plot (or, if None, will just plot
			whatever is inside the class already)

		@param plot_processed: Whether or not to plot the pre-processing tree
			or the post-processing tree
		"""
		if data is not None:
			if data.__class__ == str:
				data = ast.parse(data)

			return NodeGrapher(data).draw_graph()

		if self.stored_tree is None:
			raise ValueError("No Data to Plot!")

		if plot_processed:
			if self.stored_tree[3] is not None:
				return NodeGrapher(self.stored_tree[3]).draw_graph()
		else:
			if self.stored_tree[2].__class__ == str:
				return NodeGrapher(ast.parse(self.stored_tree[2])).draw_graph()
			else:
				return NodeGrapher(self.stored_tree[2]).draw_graph()

	def process(self, data=None):
		"""
		Processes the stored data in the class.

		Also marks everything as processed for good measure.

		@param data: The data to process (will overwrite what is currently in
			the class)
		"""
		if data is not None:
			self.stored_tree = ['Generic Function', 'Standard Input', data, None]

		if self.stored_tree is None:
			return None
		else:
			self.stored_tree[0] += " (Processed)"

			if self.stored_tree[2].__class__ == str:
				self.stored_tree[3] = ast.fix_missing_locations(self.visit(ast.parse(self.stored_tree[2])))
			else:
				self.stored_tree[3] = ast.fix_missing_locations(self.visit(self.stored_tree[2]))

			if self.print_trace:
				print "HELLO: Finished Parsing"

			return self.stored_tree[3]

	def eval(self, data=None):
		#output = ast.fix_missing_locations(AssignNumbers(self.process(data)).tree)
		#output.lineno = 0
		output = self.process(data)
		try:
			exec(compile(output, filename='<ast>', mode='exec'))
			return output
		except TypeError as e:
			print "FAILEDDDDDD"
			print e
			return output

	def __repr__(self):
		"""
		Pretty printing!
		"""
		if self.stored_tree is None:
			return "Instance of NameReplacer with no Stored Data"
		else:
			output = "<<< Instance of NameReplacer >>>\n"
			output += "       Function Name: %s\n" % self.stored_tree[0]
			output += "       File Name: %s\n" % self.stored_tree[1]
			output += "       Text Length: %d\n" % len(self.stored_tree[2])
			return output

	def __str__(self):
		"""
		See __repr__ - more pretty printing!
		"""
		return self.__repr__()

	def __randString(self, size=20, chars=string.ascii_letters):
		"""
		Generates a random string of size size from the character set
		that is specified.

		@param size: The number of characters in the random string

		@param chars: The character set in the random string
		"""
		return "".join(random.choice(chars) for _ in range(size))

	def visit_FunctionDef(self, node):
		"""
		In case we have a class definition instead of a function one
		"""
		# Visit all child nodes
		self.generic_visit(node)

		if self.print_trace:
			print "Storing Function Definition"

		argnode = node.args
		for name, call in self.new_kwargs:
			argnode.args.append(name)
			argnode.defaults.append(call)

		self.new_kwargs = []

		return node

	def visit_Return(self, node):
		# Visit all children
		self.generic_visit(node)

		logmass = []
		for k,v in self.discretevars.iteritems():
			print("We can call log likelihood here.")
			print("Not implemented yet. But this means that we removed a discrete func.")
			#logmass.append(v.logmass())

		# Use enumerate syntax for returning logmass
		if len(logmass) > 0:
			return ast.copy_location(ast.Tuple((ast.List(logmass, ast.Load()), node), ast.Load), node)

		# If no logmass, just return and go home
		return node

	def visit_Call(self, node):
		"""
		This is what is called when we attempt to make a function call.

		The transformation that we are attempting is to see if
			(1) The call has no variables
			(2) The call is not stochastic

		We have no way of determining (2) (even running many times isn't
		guaranteed to work). So, we ask the programmer to specify, both
		in the file and at runtime when the class is instantiated.
		"""
		# Enforce
		self.generic_visit(node)

		# Check to see if the arguments are all nums
		all_numeric = True

		# Caught arguments
		for a in node.args:
			if not isinstance(a, ast.Num):
				all_numeric = False

		# Caught kwargs
		for ka in node.keywords:
			if not isinstance(ka.value, ast.Num):
				all_numeric = False

		# *args
		if node.starargs is not None:
			for sa in node.starargs:
				if not isinstance(a, ast.Num):
					all_numeric = False

		# **kwargs
		if node.kwargs is not None:
			for ska in node.kwargs:
				if not isinstance(ska.value, ast.Num):
					all_numeric = False

		# There are some variables still left in the call.
		if not all_numeric:
			return node

		# Not super tested yet >.>
		if isinstance(node.func, ast.Name):
			name = node.func.id
		elif instance(node.func, ast.Attribute):
			name = node.func.attr
		else:
			return node

		# Make sure function is deterministic
		if name not in self.stochastic:
			if self.eval_funcs:
				v = eval(compile(ast.fix_missing_locations(ast.Expression(node)), "<ast>", "eval"))

				# Proper return type tested for: functions, vars, lists, classes
				# Untested for rest - should test at some point.
				return v.body[0].value # Returned should be an expression
			else:
				# Cache in class for loading at compile time
				newname = ast.fix_missing_locations(ast.Name(self.__randString(), ast.Load()))
				self.new_kwargs.append([ast.fix_missing_locations(ast.Name(newname.id, ast.Param())), node])
				return newname
		else:
			return node


	def visit_Assign(self, node):
		"""
		Runs on assignment nodes. Takes away the first instance of
		any variable that we are setting (on the left hand side only),
		and replaces through the rest of the code.

		Will keep up with variable updating; that is, if a = a+2, then
		the replacer will know to reset a to two more than it was.
		"""
		if self.print_trace:
			print "VISIT ASSIGN"
			print node.value
			print self.already_done
			print node.targets

		te = node.targets[0]

		if isinstance(te, ast.Attribute):
			internal_name = te.value.id + "." + te.attr
		elif isinstance(te, ast.Name):
			internal_name = te.id
		else:
			internal_name = None

		if internal_name is not None:
			if internal_name in self.already_done and not self.already_done[internal_name]:
				self.already_done[internal_name] = True
				if isinstance(node.value, ast.Call):
					if isinstance(node.value.func, ast.Attribute):
						func_name = node.value.func.value.id + "." + node.value.func.attr
					elif isinstance(te, ast.Name):
						func_name = node.value.func.id
					else:
						func_name = None

					if (func_name is not None) and func_name in self.stochastic:
						self.discretevars[func_name] = node.value
				return None

		self.generic_visit(node)

		# Single assignment case
		if isinstance(te, ast.Name):
			# Call the recursion on the right hand side ONLY
			# self.generic_visit(node)
			try:
				if self.already_done[te.id]:
					if isinstance(node.value, ast.Num):
						self.replace[te.id] = node.value.n
						return None
					else:
						return node
				else:
					# We don't want to assign anymore
					self.already_done[te.id] = True
					return None
			except KeyError:
				# Remove useless statements now
				if isinstance(node.value, ast.Num):
					self.replace[te.id] = node.value.n
					self.already_done[te.id] = True
					return None
				
				if isinstance(node.value, ast.List):
					try:
						self.replace[te.id] = [e.n for e in node.value.elts]
						self.already_done[te.id] = True
						self.non_numeric[te.id] = True
						return None
					except AttributeError:
						pass
				return node

		# Multi-Assignment case
		# Catch the instance of >> x,y = [1,2]
		if isinstance(te, ast.Tuple):
			# Three cases
			if isinstance(node.value, ast.Num):
				raise ValueError("Expression Doesn't Compile! (TODO: Better Errors)")

			if isinstance(node.value, ast.Tuple) or isinstance(node.value, ast.List):
				if len(te[0].elts) != len(node.value.elts):
					raise ValueError("Expression Still Doesn't Compile!")

				for i,e in enumerate(te[0].ets):
					if isinstance(e, ast.Name):
						if e.id in self.already_done:
							if self.already_done[e.id]:
								self.replace[e.id] = node.value.elts[i]

							# Remove the assignment
							node.targets[0].elts[i] = None
							node.value.elts[i] = None
							return node
						else:
							# TODO: IMPLEMENT ADDING TO DICT TO DECREASE COMPUTATION
							pass
					else:
						# TODO: FINISH CODING THIS WHEN NEEDED.
						raise NotImplementedError("Nested Multi-Assignment Not Yet Supported!")

		# Array index case
		if isinstance(te, ast.Subscript):
			if isinstance(node.value, ast.Num):
				self.replace[te.value.id][te.slice.value.n] = node.value.n
				return None
			else:
				# replace the instruction with the original instruction
				astvals = [ast.fix_missing_locations(ast.Num(e)) for e in self.replace[te.value]]
				astvals[te.slice] = node.value

				del self.replace[te.value.id]
				del self.already_done[te.value.id]

				# New instruction replacement
				return ast.copy_location(ast.Assign(ast.Name(te.value.id, ast.Store()), ast.List(astvals, ast.Load())), node)
	

		# Class Attribute Case
		if isinstance(te, ast.Attribute):
			# Call the recursion on the right hand side ONLY
			# self.generic_visit(node)
			internal_name = te.value.id + "." + te.attr
			try:
				if self.already_done[internal_name]:
					# check right hand side
					if isinstance(node.value, ast.Num):
						# If instant evaluation is possible
						self.replace[internal_name] = node.value.n
						return None
					else:
						return node
				else:
					# We don't want to assign anymore
					self.already_done[internal_name] = True
					return None
			except KeyError:
				# Remove useless statements now
				if isinstance(node.value, ast.Num):
					self.replace[internal_name] = node.value.n
					self.already_done[internal_name] = True
					return None
				
				if isinstance(node.value, ast.List):
					try:
						self.replace[internal_name] = [e.n for e in node.value.elts]
						self.already_done[internal_name] = True
						self.non_numeric[internal_name] = True
						return None
					except AttributeError:
						pass

				return node

		return node

	def visit_Attribute(self, node):
		"""
		For visiting variable attribute nodes.
		"""
		if self.print_trace:
			print "VISIT ATTRIBUTE"

		internal_name = node.value.id + "." + node.attr
			
		if internal_name in self.replace:
			if internal_name not in self.non_numeric:
				if isinstance(node.ctx, ast.Load):
					if self.print_trace:
						print "VISITED %s" % internal_name
					return ast.copy_location(ast.Num(self.replace[internal_name]), node)

				if isinstance(node.ctx, ast.Param):
					if self.print_trace:
						print "PARAM %s" % internal_name
					self.already_done[internal_name] = True
					return None
		return node

	def visit_Subscript(self, node):
		"""
		Works for list indexing. See above for what this is doing.
		"""
		if self.print_trace:
			print "VISIT SUBSCRIPT"
		self.generic_visit(node)
		if node.value.id in self.replace:
			if isinstance(node.ctx, ast.Load) and isinstance(node.slice.value, ast.Num):
				try:
					if self.print_trace:
						print "REPLACING"
						print node.value.id
						print node.slice.value.n
						print self.replace[node.value.id]
					return ast.copy_location(ast.Num(self.replace[node.value.id][node.slice.value.n]), node)
				except:
					raise ValueError("Something went wrong in the subscripting code.")

			if isinstance(node.ctx, ast.Store) and isinstance(node.slice.value, ast.Num):
				self.replace[node.value.id][node.slice.value.n] = 0
		return node

	def visit_Name(self, node):
		"""
		For visiting variable name nodes.
		"""
		if self.print_trace:
			print "VISIT NAME"
			
		if node.id in self.replace:
			if node.id not in self.non_numeric:
				if isinstance(node.ctx, ast.Load):
					if self.print_trace:
						print "VISITED %s" % node.id
					return ast.copy_location(ast.Num(self.replace[node.id]), node)

				if isinstance(node.ctx, ast.Param):
					if self.print_trace:
						print "PARAM %s" % node.id
					self.already_done[node.id] = True
					return None
		return node

	def visit_BinOp(self, node):
		"""
		For visiting binary operation nodes.
		"""
		if self.print_trace:
			print "VISIT BINOP"
		self.generic_visit(node)
		try:
			if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
				val = eval(compile(ast.fix_missing_locations(ast.Expression(node)), "<ast>", "eval"))
				if self.print_trace:
					print "REPLACED:"
				return ast.copy_location(ast.Num(val), node)
		except AttributeError:
			print "Node failed to return."
			print node
			for f in node._fields:
				print getattr(node, f)
		return node

class AssignNumbers(ast.NodeVisitor):
	def __init__(self, tree):
		super(AssignNumbers, self).__init__()
		self.lineno = 0

		try:
			if tree.__class__ == str:
				self.tree = ast.parse(tree)
			else:
				self.tree = tree
			self.generic_visit(self.tree)
		except TypeError as e:
				raise TypeError("Input Type Incorrect - either tree or string")

	def generic_visit(self, node):
		for attr, vs in ast.iter_fields(node):
			if isinstance(vs, ast.AST):
				vs.lineno = self.lineno
				vs.col_offset = 0
				self.lineno += 1
				return self.visit(vs)
			elif isinstance(vs, list):
				for v in vs:
					if isinstance(v, ast.AST):
						v.lineno = self.lineno
						v.col_offset = 0
						self.lineno += 1
						return self.visit(v)
			else:
				pass


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
		self.ax.cla()
		root = self.__closest_point(event.xdata, event.ydata)

		graph = copy.deepcopy(self.graph)

		dict_delete = []
		for k,vs in graph.iteritems():
			try:
				if (self.pos[k][1] >= event.ydata) and (k != root):
					dict_delete.append(k)
				else:
					to_delete = []
					for i in xrange(len(vs)):
						if self.pos[vs[i]][1] >= event.ydata:
							to_delete.append(i)

					for i in reversed(to_delete):
						del vs[i]
			except KeyError:
				dict_delete.append(k)

		for k in dict_delete:
			del graph[k]

		self.G = nx.Graph(graph)
		return self.__dfs_plot(root, graph, {k:None for k in graph.iterkeys()})

	def __closest_point(self, xval, yval):
		try:
			allpos = self.pos.items()
			dists = [np.abs(x-xval)/10.+np.abs(y-yval) for _, (x, y) in allpos]
			return allpos[np.argmin(dists)][0]
		except AttributeError as e:
			raise AttributeError("Draw was not called first!")

	def __dfs_plot(self, root, graph=None, levels=None):
		if levels is None:
			levels = {k:None for k in self.labels.iterkeys()}

		if graph is None:
			graph = self.graph

		levels[root] = (0,0,1)
		queue = [root]

		while len(queue) > 0:
			val = queue.pop()
			h, minval, maxval = levels[val]
			if len(graph[val]) > 0:
				diff = float(maxval-minval)/len(graph[val])
				for i,v in enumerate(graph[val]):
					try:
						if levels[v] is None:
							levels[v] = (h+1,minval+i*diff, minval+(i+1)*diff)
							queue.append(v)
					except KeyError:
						pass
					
		diff = 2./(1+len([1 for v in levels.itervalues() if v is None]))
		#self.pos = {k: (diff*(1+1),1) if v is None else (v[1]+v[2], -1*v[0]) for k,v in levels.iteritems()}
		self.pos = {k:(diff,1) if v is None else (v[1]+v[2], -1*v[0]) for k,v in levels.iteritems()}

		if self.ax is None:
			self.fig, self.ax = plt.subplots(1)

		nx.draw(self.G, self.pos, node_color='w', node_shape='', \
			width=3., node_size = 600, alpha=0.1, ax=self.ax)
		nx.draw_networkx_labels(self.G, self.pos, labels={k:self.labels[k] for k,v in levels.iteritems() if v is not None}, ax=self.ax)

		self.fig.canvas.mpl_connect('button_press_event', lambda event:self.onclick(event))

		plt.show(block=False)

		return self.fig