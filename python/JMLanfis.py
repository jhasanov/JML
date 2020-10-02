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
		return (1/(1+math.pow(math.fabs((x-a+0.000000001)/c),2*b)))
		
	def calculateOut(self,x,p):
		if self.MF == 'SIGMOID':
			return self.sigmoid(x,p[0])
		elif self.MF == 'BELL':
			return self.bell(x,p[0],p[1],p[2])
		elif self.MF == 'CBELL':
			return self.bell(x,0.0,p[0],p[1])
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
#Input1
a = Activation([1.209531704166206],'SIGMOID',0)
activations.append(a)
a = Activation([-1.194465546633537],'SIGMOID',0)
activations.append(a)
a = Activation([-1.340091917541952,-0.6271021494657256],'CBELL',0)
activations.append(a)
a = Activation([0.9398003977236826,0.5220991289081719,1.8645707044576543],'BELL',0)
activations.append(a)
a = Activation([-1.093463857026576],'SIGMOID',0)
activations.append(a)
#Input2
a = Activation([-0.3393782698154289],'SIGMOID',1)
activations.append(a)
a = Activation([0.6454608284193266],'SIGMOID',1)
activations.append(a)
a = Activation([-1.4561343122408998,-0.2358874974785893],'CBELL',1)
activations.append(a)
a = Activation([0.9673980836727998,-1.5922906352584083,-0.5944322498240807],'BELL',1)
activations.append(a)
a = Activation([3.0896571018569787],'SIGMOID',1)
activations.append(a)
#Input3
a = Activation([-3.2179167170684466],'SIGMOID',2)
activations.append(a)
a = Activation([-2.512308690535301],'SIGMOID',2)
activations.append(a)
a = Activation([2.7468648657651147,-0.5872016657950152],'CBELL',2)
activations.append(a)
a = Activation([-1.7098420155921912,-0.30640889890176964,0.32070416008351493],'BELL',2)
activations.append(a)
a = Activation([3.1344332694511388],'SIGMOID',2)
activations.append(a)
#Input4
a = Activation([0.4670150061826003,-0.8048144849968274],'CBELL',3)
activations.append(a)
a = Activation([-1.3870317045443306,-0.6568535278381032,9.378855304968107E-5],'BELL',3)
activations.append(a)
a = Activation([1.3842191218411586],'SIGMOID',3)
activations.append(a)
a = Activation([-1.5889426928250774],'SIGMOID',3)
activations.append(a)
a = Activation([0.5802851810539001,-0.3434894929582596],'CBELL',3)
activations.append(a)
#Input5
a = Activation([-1.3493667712327637,0.844330727248313],'CBELL',4)
activations.append(a)
a = Activation([0.6876575484770804,-1.0848707533904631,0.87990983323952],'BELL',4)
activations.append(a)
a = Activation([1.828095686579726],'SIGMOID',4)
activations.append(a)
a = Activation([-0.3853542197383324],'SIGMOID',4)
activations.append(a)
a = Activation([0.6181761687102579,0.25581850776691833],'CBELL',4)
activations.append(a)
#Input6
a = Activation([0.06287755486834931,-0.23873260143588138],'CBELL',5)
activations.append(a)
a = Activation([-0.6594275681012514,1.030990523035627,0.41115052825548487],'BELL',5)
activations.append(a)
a = Activation([-4.296953732196322],'SIGMOID',5)
activations.append(a)
a = Activation([2.406043634629037],'SIGMOID',5)
activations.append(a)
a = Activation([0.66690923699417,-0.33776797892279636],'CBELL',5)
activations.append(a)

# Rule layer
rules = []
r = Rule([0, 5, 15, 20, 25],'AND')
rules.append(r)
r = Rule([1, 6, 16, 21, 26],'AND')
rules.append(r)
r = Rule([2, 7, 17, 22, 27],'AND')
rules.append(r)
r = Rule([3, 8, 18, 23, 28],'AND')
rules.append(r)
r = Rule([4, 9, 19, 24, 29],'AND')
rules.append(r)
r = Rule([0, 10, 15, 20, 25],'AND')
rules.append(r)
r = Rule([1, 11, 16, 21, 26],'AND')
rules.append(r)
r = Rule([2, 12, 17, 22, 27],'AND')
rules.append(r)
r = Rule([3, 13, 18, 23, 28],'AND')
rules.append(r)
r = Rule([4, 14, 19, 24, 29],'AND')
rules.append(r)
r = Rule([10, 5, 15, 20, 25],'AND')
rules.append(r)
r = Rule([11, 6, 16, 21, 26],'AND')
rules.append(r)
r = Rule([12, 7, 17, 22, 27],'AND')
rules.append(r)
r = Rule([13, 8, 18, 23, 28],'AND')
rules.append(r)
r = Rule([14, 9, 19, 24, 29],'AND')
rules.append(r)


# Consequent parameters
conseqParams = []
conseqParams.append(-0.5911622015523783)
conseqParams.append(0.8295708952685849)
conseqParams.append(-1.1189867134747318)
conseqParams.append(0.46031090557352566)
conseqParams.append(0.996820743262786)
conseqParams.append(2.20593695624828)
conseqParams.append(0.47906939294606704)
conseqParams.append(-0.6158525053552107)
conseqParams.append(0.8947546369167159)
conseqParams.append(0.9278730865577803)
conseqParams.append(0.4572748224259176)
conseqParams.append(-0.9062874989177515)
conseqParams.append(0.3394323838461775)
conseqParams.append(0.5876938792717107)
conseqParams.append(0.6754494971186276)
conseqParams.append(0.16568031451015727)
conseqParams.append(0.6050284000114061)
conseqParams.append(-0.3062108925736443)
conseqParams.append(0.30236066702554937)
conseqParams.append(-1.4742289420644623)
conseqParams.append(-0.1586442115352871)
conseqParams.append(0.5863949545772119)
conseqParams.append(-1.6173047870360164)
conseqParams.append(1.3738986281001828)
conseqParams.append(-1.2245050585882498)
conseqParams.append(1.2558317051719547)
conseqParams.append(-0.35415548178426354)
conseqParams.append(-0.08511634093924676)
conseqParams.append(-0.5528277416507243)
conseqParams.append(1.5139238148319436)
conseqParams.append(0.049848127248001695)
conseqParams.append(1.0140055989249903)
conseqParams.append(-2.5713124148802193)
conseqParams.append(-0.47214507746275985)
conseqParams.append(0.8953947382918055)
conseqParams.append(-0.7217407102535343)
conseqParams.append(0.311181157224318)
conseqParams.append(0.22579248415212372)
conseqParams.append(0.40110553553346406)
conseqParams.append(1.4540613343539377)
conseqParams.append(1.7367105760729566)
conseqParams.append(-1.0160457642656253)
conseqParams.append(0.25602942880729673)
conseqParams.append(0.5471549462370359)
conseqParams.append(0.32788465999145544)
conseqParams.append(1.3588803042197746)
conseqParams.append(-1.4565463246506036)
conseqParams.append(0.30046242593882216)
conseqParams.append(0.6536010492590593)
conseqParams.append(-0.43502447005609657)
conseqParams.append(1.093370935622958)
conseqParams.append(0.2898937555874575)
conseqParams.append(-1.5696638282167956)
conseqParams.append(-2.560416261890163)
conseqParams.append(1.2870061244180016)
conseqParams.append(1.0235704791073552)
conseqParams.append(-0.20260343992099852)
conseqParams.append(0.16664568311006095)
conseqParams.append(-0.5202305714639143)
conseqParams.append(-1.1110903976366457)
conseqParams.append(3.2094345276107292)
conseqParams.append(0.9797831754882353)
conseqParams.append(0.06440960676632375)
conseqParams.append(0.524919446647031)
conseqParams.append(-0.4423758061053973)
conseqParams.append(0.7594895214047954)
conseqParams.append(0.29601430525701594)
conseqParams.append(1.4234284814846367)
conseqParams.append(-0.7635516838317128)
conseqParams.append(0.9247761234484801)
conseqParams.append(-0.7945706085702757)
conseqParams.append(-0.48718673907008353)
conseqParams.append(-0.16683134632552565)
conseqParams.append(0.9513914054240082)
conseqParams.append(0.6479628726973753)
conseqParams.append(1.9871981091185797)
conseqParams.append(-0.9731613543406384)
conseqParams.append(-0.4310921333214293)
conseqParams.append(0.5200955586104671)
conseqParams.append(0.7311459237402672)
conseqParams.append(1.1829183481408467)
conseqParams.append(-1.419194570921276)
conseqParams.append(-0.07988868158679811)
conseqParams.append(0.009350284640309521)
conseqParams.append(-0.716454956932292)
conseqParams.append(0.4259900838120243)
conseqParams.append(-0.7265922943211856)
conseqParams.append(0.4892015691196988)
conseqParams.append(-1.3816330614630918)
conseqParams.append(-0.17028692411068774)
conseqParams.append(0.7498615445214688)
conseqParams.append(0.8671525444855672)
conseqParams.append(-1.0397305325487403)
conseqParams.append(1.3099406677705718)
conseqParams.append(-0.2130807428309335)
conseqParams.append(2.069372545138716)
conseqParams.append(-2.141873948435138)
conseqParams.append(0.06602434784259541)
conseqParams.append(1.214264722447564)
conseqParams.append(1.3969746323961472)
conseqParams.append(2.497683707007403)
conseqParams.append(0.2206556009014594)
conseqParams.append(1.045501671420323)
conseqParams.append(1.9817528337357473)
conseqParams.append(2.0102439555175344)

# Run ANFIS
#anfisOut = calculateAnfisOutput([0.882320595,0.563664832,0.598688422,-0.093001396,0.123553977,-0.196029129])
#print('ANFIS Output=',anfisOut) # Expected: 0.7781891216877017
#anfisOut = calculateAnfisOutput([0.563295427,0.942495955,0.318241689,0.02602864,7.77329E-4,0.031191507])
#print('ANFIS Output=',anfisOut) # Expected: 0.972983662428136
#anfisOut = calculateAnfisOutput([0.686030132,0.734109198,0.426947117,0.0,0.0,0.0])
#print('ANFIS Output=',anfisOut) # Expected: 0.9986566212142745

# Test ANFIS