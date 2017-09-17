import math

activations = []
rules = []
conseqParams = []

class Activation:
	"This class helps create an input layer of ANFIS"
	def __init__(self,p,MF):
		self.params = p
		self.MF = MF
		self.input = 0.0
		self.output = 0.0
		
	def setInput(self,x):
		self.input = x
		self.output = self.calculateOut(x,self.params)
		
	def sigmoid(self,x,a):
		return (1 / (1 + math.exp(a*x)))
		
	def bell(self,x,a,b,c):
		return (1/(1+math.pow(math.abs((x-a)/c),2*b)))
		
	def calculateOut(self,x,p):
		if self.MF == 'SIGMOID':
			return self.sigmoid(x,p[0])
		elif self.MF == 'BELL':
			return self.bell(x,p[0],p[1],p[2])
		else: 
			return 0
	
class Rule:
	"This is a rule element in ANFIS"
	def __init__(self, idx, ruleOper):
		self.inputIndex = idx
		self.operation = ruleOper
		self.output = 0.0
		
	def calcOut(self,inputs):
		if self.operation == 'AND':
			self.output = 1.0
			
			for i in self.inputIndex:
				self.output *= inputs[i]
		elif self.operation == 'OR':
			for i in self.inputIndex:
				self.output = math.max(self.output,inputs[i])
		else:
			self.output = 0.0
		
def normalize(inputIdx,inputs):
	norm = 0.0;
	for i in inputs:
		norm += i
	return norm / inputs[inputIdx]
	
def calculateAnfisOutput(inputVals):
	ruleInputs = []
	for i,a in enumerate(activations):
		a.setInput(inputVals[i])
		ruleInputs.append[a.output]
	
	normInputs = []
	for i,r in enumerate(rules):
		r.calcOut(ruleInputs)
		normInputs[i] = r.output
		
	normOutput = []	
	for i,n in enuamerate(normInputs):
		normOutput[i] = normalize(i,normInputs)
	
	anfisOut = 0.0
	for i in range(0,len(conseqParams)):
		anfisOut +=

	return anfisOut

inputs = []
a = Activation([1],'SIGMOID')
inputs.append(a)
inputs[0].setInput(1)
print('Output=',inputs[0].output)
