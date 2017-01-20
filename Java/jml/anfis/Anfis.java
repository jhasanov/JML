package jml.anfis;

public class Anfis {
	double minError = 0.001;
	int maxIterCnt = 100;
	
	
	int inputCnt = 4;
	int activationCnt = 6;

	Activation[] activationList;
	Rule[] ruleList;
	
	double [] normalizedVals;
	double [] defuzzVals;
	double outputVal;
	
	double [] membershipParams;
	double [] linearParams;
		
	public Anfis() {
		
	}	
	
	public void initialize() {
		// setup activations
		activationList = new Activation[6];
		activationList[0] = new Activation(0,Activation.MembershipFunc.SIGMOID);
		activationList[1] = new Activation(0,Activation.MembershipFunc.GAUSS);
		activationList[2] = new Activation(1,Activation.MembershipFunc.SIGMOID);
		activationList[3] = new Activation(1,Activation.MembershipFunc.GAUSS);
		activationList[4] = new Activation(2,Activation.MembershipFunc.GAUSS);
		activationList[5] = new Activation(3,Activation.MembershipFunc.GAUSS);
		
		// setup rules
		ruleList = new Rule[4];
		ruleList[0] = new Rule((new int[]{0,2,3,4}), Rule.RuleOperation.AND);
		ruleList[1] = new Rule((new int[]{1,3,3,4}), Rule.RuleOperation.AND);
		ruleList[2] = new Rule((new int[]{1,2,3,4}), Rule.RuleOperation.AND);
		ruleList[3] = new Rule((new int[]{0,3,3,4}), Rule.RuleOperation.AND);
		
		normalizedVals = new double[ruleList.length];
		defuzzVals = new double[ruleList.length];
		// +1 bias parameter
		linearParams = new double[inputCnt+1];
		for (int i=0;i<linearParams.length; i++) 
			linearParams[i] = 0.5;
	}

	double runForward(double [] inputs) {
		// calculate Activation values
		for (int i = 0; i< activationList.length; i++) {
			activationList[i].activate(inputs);
		}
		
		// calculate Rules and total sum of them (for normalization phase)
		double ruleSum = 0.0;
		for (int i = 0;i < ruleList.length; i++) {
			ruleSum += ruleList[i].calculate(activationList);
		}
		
		// Normalize - iterate through rules and calculate normalized value
		for (int i=0; i<normalizedVals.length; i++) {
			normalizedVals[i] = ruleList[i].getRuleVal()/ruleSum;
		} 
		
		// Defuzzification
		for (int i=0; i<defuzzVals.length; i++) {
			for (int j=0; j<inputs.length; j++) {
				defuzzVals[i] += linearParams[j]*inputs[j];			
			}
			// add bias parameter 
			defuzzVals[i] += linearParams[linearParams.length-1];
			defuzzVals[i] *= inputs[i];
		}

		// Summations
		for (int i = 0; i<defuzzVals.length; i++) {
			outputVal += defuzzVals[i];
		}
		
		return outputVal;
	}
	
	void calculateLinear() {
		
	}
	
	void calculateFuzzy() {
		
	}

	public double getOutputVal() {
		return outputVal;
	}
	
	public void runLearning() {
		int iterCnt = 0;
		
		while ((err > minError) && (iterCnt++ < maxIterCnt) ) {
			//runForward();
			calculateLinear();
			calculateFuzzy();
		}
	}
	
	public static void main (String [] args) {
		Anfis anfis = new Anfis();
		anfis.initialize();
		anfis.runForward(new double[]{0.1, 0.2,-0.5,0.5});
		System.out.println("Output: "+anfis.getOutputVal());
	}
}

class Input {
	Activation activation;
}

class Activation {
	enum MembershipFunc { SIGMOID, GAUSS};

	MembershipFunc mf;
	double [] params;
	int inputNo; // index of the input
	
	double activationVal = 0.0;
		
	public Activation(int inputNo, MembershipFunc mf) {
		this.inputNo = inputNo;
		this.mf = mf;

		if (mf == MembershipFunc.SIGMOID) {
			params = new double[1];
			params[0] = 0.5;
		}
		else if (mf == MembershipFunc.GAUSS) {
			params = new double[2];
			params[0] = 0.5;
			params[1] = 0.5;
		}
	}
	
	double sigmoidFunc(double x,double a) {
		return (1/(1+Math.exp(a*x)));
	}
	
	double gaussFunc(double x,double a,double b) {
		return Math.exp(-1*(Math.pow((x-a),2))/(2*Math.pow(b,2)));
	}

	public double activate(double[] inputs) {
		if (mf == MembershipFunc.SIGMOID)
			activationVal = sigmoidFunc(inputs[inputNo],params[0]);
		else if (mf == MembershipFunc.GAUSS) 
			activationVal = gaussFunc(inputs[inputNo], params[0],params[1]);
		else
			activationVal = -1.0;
		
		return activationVal;
	}
	
	public double getActivationVal() {
		return activationVal;
	}
}

class Rule {
	enum RuleOperation {AND, OR};
	
	RuleOperation oper;
	int [] inputActivations;
	double ruleVal;
	
	public Rule(int [] inputMFidx, RuleOperation oper) {
		this.inputActivations = inputMFidx;
		this.oper = oper;
	}
	
	public double calculate(Activation [] mfInputs) {
		if (oper == RuleOperation.AND) {
			ruleVal = 1.0;
			
			for (int i = 0; i< inputActivations.length; i++) {
				ruleVal = ruleVal * mfInputs[inputActivations[i]].getActivationVal();
			}
		}
		else if (oper == RuleOperation.OR) {
			ruleVal = 1.0;
			
			for (int i = 0; i< inputActivations.length; i++) {
				ruleVal = Math.min(ruleVal, mfInputs[inputActivations[i]].getActivationVal());
			}
		}
		
		return ruleVal;
	}

	public double getRuleVal() {
		return ruleVal;
	}
	
}