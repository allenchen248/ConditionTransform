import ast
import networkx as nx
from util.ast import NameReplacer, NodeGrapher

# Return class with function/logmass or return two things (w/ logmass)
# Two weeks from now drop by

doublereturn = """def dr(x,y):
	z = x+y
	return z,x*y"""

tree = """class TestNormal(Dist):
    def __init__(self):
        self.discrete = Discrete([1,1],size=1)
    def run(self,trace):
        t = trace.chimplify("choice", self.discrete)
        return trace.chimplify("x",Normal(0,t*10,1))"""

# Parses this into: (with a=5)
# def test_func():
#      return 16
def basic_test():
	"""
	Correct output (success!):
		def test_func():
			return 16
	"""
	tree = ast.parse(
	"""def test_func(a):
		a = a+4
		b = 2+5
		return a+b""")
	te = NameReplacer(a=5).process(tree)
	assert(te.body[0].body[0].value.n == 16)

def param_test():
	"""
	Correct output (success!):
		def test_func(c):
			return 16+c
	"""
	tree = ast.parse(
	"""def test_func(a, c):
		a = a+4
		b = 2+5
		return a+b+c""")
	te = NameReplacer(a=5)
	te.process(tree)
	return te

def kwarg_test():
	tree = ast.parse(
	"""def test_func(a, c=10, d=20):
		a = a+4
		b = 2+5
		return a+b+c""")
	te = NameReplacer(a=5)
	te.process(tree)
	return te

def subscript_test():
	"""
	Correct output (success!):
		def test_func():
			return 13
	"""
	tree = ast.parse(
	"""def test_func(a):
		xs = [0,10,20,30,40]
		return a+xs[a]""")
	te = NameReplacer(a=3)
	te.process(tree)
	return te

def list_test():
	"""
	Correct output (success!):
		def test_func():
			return 103
	"""
	tree = ast.parse(
	"""def test_func(a):
		xs = [0,10,20,30,40]
		xs[a] = 100
		return a+xs[a]""")
	te = NameReplacer(a=3)
	te.process(tree)
	return te

def function_test():
	"""
	Seems to work!
	"""
	def useless_func(a):
		print "YOU'VE CALLED A USELESS FUNCTION WITH VALUE %d" % a
		return a+10

	tree = """def test_func(a):
			return useless_func(a)"""

	truth = ast.parse(
		"""def test_func(wxs=useless_func(5)):
			return wxs""")

	print "Processing"
	te = NameReplacer(a=5)
	res = te.process(tree)

	# Should print
	print "Compiling"
	exec(compile(ast.fix_missing_locations(res), '<ast>', 'exec'))

	# Should NOT print (3 times)
	print "Running"
	assert(test_func() == 15)
	assert(test_func() == 15)
	assert(test_func() == 15)

	return te

def viz_test():
	te = NodeGrapher(ast.parse("""def test_func(a):
		xs = [0,10,20,30,40]
		xs[a] = 100
		return a+xs[a]"""))
	te.draw_graph()

def test_func(a):
	a = a+4
	b = 2+5
	return a+b