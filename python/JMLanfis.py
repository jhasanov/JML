import math

activations = []
rules = []
conseqParams = []

class Activation:
	"This class helps create an input layer of ANFIS"
	def __init__(self,p,MF,inputIdx):
		self.params = p
		self.MF = MF
		self.input = 0.0
		self.output = 0.0
		self.inputIdx = inputIdx
		
	def setInput(self,x):
		self.input = x[self.inputIdx]
		self.output = self.calculateOut(self.input,self.params)
		
	def sigmoid(self,x,a):
		return (1 / (1 + math.exp(a*x)))
		
	def bell(self,x,a,b,c):
		return (1/(1+math.pow(math.fabs((x-a)/c),2*b)))
		
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
	ruleSum = 0.0;
	for i in inputs:
		ruleSum += i
	return inputs[inputIdx] / ruleSum 
	
def calculateAnfisOutput(inputVals):
	# Calculate the output of the input layer
	mfOutput = []
	for i,a in enumerate(activations):
		a.setInput(inputVals)
		mfOutput.append(a.output)
	
	# Calculate the output of the rule layer
	normInputs = []
	for i,r in enumerate(rules):
		r.calcOut(mfOutput)
		normInputs.append(r.output)
		
	# Normalization of the rule outputs
	normOutput = []	
	for i,n in enumerate(normInputs):
		normOutput.append(normalize(i,normInputs))
	
	# Defuzzification 
	defuzzOut = []
	for i in range(0,len(normOutput)):
		defuzzOut.append(0.0)
		for j in range (0,len(inputVals)):
			defuzzOut[i] += conseqParams[i * (len(inputVals)+1) + j] * inputVals[j]
		defuzzOut[i] += conseqParams[i * (len(inputVals) + 1) + len(inputVals)]
		defuzzOut[i] *= normOutput[i]
    
	# Output
	anfisOut = 0.0
	for i in range (0,len(defuzzOut)):
		anfisOut += defuzzOut[i];

	return anfisOut

#Initialize ANFIS 

# Activation layer
activations = []
a = Activation([0.2526070131836236],'SIGMOID',0)
activations.append(a)
a = Activation([0.33681599970387666,0.25541903708925434,-0.20067384874220087],'BELL',0)
activations.append(a)
a = Activation([0.37609416384438976],'SIGMOID',1)
activations.append(a)
a = Activation([0.29864903126610964,0.9931658281943053,0.9792534054217865],'BELL',1)
activations.append(a)
a = Activation([0.2208031549412901],'SIGMOID',2)
activations.append(a)
a = Activation([0.4258796688714126,0.47919428847349743,0.6168753686001132],'BELL',2)
activations.append(a)

# Rule layer
rules = []
r = Rule([0,2,4],'AND')
rules.append(r)
r = Rule([1,3,5],'AND')
rules.append(r)

# Consequent parameters
conseqParams = []
conseqParams.append(-2.741542884267103)
conseqParams.append(-19.297401950395987)
conseqParams.append(-39.47311511551012)
conseqParams.append(19.870126737995015)
conseqParams.append(1.3382976822138977)
conseqParams.append(8.662420525973562)
conseqParams.append(16.42271065616099)
conseqParams.append(-8.371405674248992)

# Run ANFIS
anfisOut = calculateAnfisOutput([0.2,0.12,0.9])
print('ANFIS Output=',anfisOut)
anfisOut = calculateAnfisOutput([0.9,0.7,0.3])
print('ANFIS Output=',anfisOut)

# Test ANFIS