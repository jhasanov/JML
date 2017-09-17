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
				print('idx=',i)
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
	return inputs[inputIdx] / norm 
	
def calculateAnfisOutput(inputVals):
	# Calculate the output of the input layer
	ruleInputs = []
	for i,a in enumerate(activations):
		a.setInput(inputVals[i])
		ruleInputs.append(a.output)
	
	# Calculate the output of the rule layer
	normInputs = []
	for i,r in enumerate(rules):
		r.calcOut(ruleInputs)
		normInputs[i] = r.output
		
	# Normalization of the rule outputs
	normOutput = []	
	for i,n in enuamerate(normInputs):
		normOutput[i] = normalize(i,normInputs)
	
	# Defuzzification 
	defuzzOut = 0.0
	for i in range(0,len(normOutput)):
		for j in range (0,len(inputs)):
			defuzzOut[i] += conseqParams(i*(len(inputs)+j)*inputs[j])
		defuzzOut[i] += conseqParams[i * (len(inputs) + 1) + len(inputs)]
		defuzzOut[i] *= normOutput[i]
    
	# Output
	for i in range (0,len(defuzzOut)):
		anfisOut += defuzzOut[i];

	return anfisOut

#Initialize ANFIS 

# Activation layer
activations = []
a = Activation([0.3469751956713573],'SIGMOID')
activations.append(a)
a = Activation([-0.3602017301249752,0.8608612820470478,1.4728785356540515],'BELL')
activations.append(a)
a = Activation([0.5057544075359199],'SIGMOID')
activations.append(a)
a = Activation([0.6680851288006711,1.404168285776265,1.0036781997949733],'BELL')
activations.append(a)
a = Activation([0.8846871777203922],'SIGMOID')
activations.append(a)
a = Activation([0.01620438496302059,0.9446804742724557,0.2155606950044786],'BELL')
activations.append(a)

# Rule layer
rules = []
r = Rule([0,2,4],'AND')
rules.append(r)
r = Rule([1,3,5],'AND')
rules.append(r)

# Consequent parameters
conseqParams = []
conseqParams.append(-0.6975957982951323)
conseqParams.append(1.7824131656273314)
conseqParams.append(12.383387901001988)
conseqParams.append(-2.243338181736341)
conseqParams.append(-0.9387706010885202)
conseqParams.append(-0.9104777162356049)
conseqParams.append(-4.198172894239717)
conseqParams.append(1.9667743362750156)

# Run ANFIS
anfisOut = calculateAnfisOutput([0.1,0.1,0.2,0.2,0.3,0.3])
print('ANFIS Output=',anfisOut)

# Test ANFIS