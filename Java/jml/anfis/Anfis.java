package jml.anfis;

import com.sun.org.apache.xalan.internal.xsltc.dom.MatchingIterator;
import jml.utils.MatrixOperations;

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
	int linearParamCnt;

    double[][] X;
    double[][][] S;
    double gamma = 1000;


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
		linearParamCnt = (inputCnt+1)*ruleList.length;
		linearParams = new double[linearParamCnt];
	}

	void start(double[][] inputs,double[] outputs) {
		// initialize S matrix: S = gamma*I
		// gamma - positive large number
		// I     - Identity matrix
        // Note: we need to keep only 2 values of S - previous (index=0) and current (index=1)
		S = new double[2][linearParamCnt][linearParamCnt];
		for (int i=0; i<S.length; i++)
			S[0][i][i] = gamma * 1.0;

		// Used to find the best linear parameters
		X = new double[2][linearParamCnt];

		// iterate over the training set
		for (int i=0; i<inputs.length; i++) {
            forwardPass(inputs[i],outputs[i]);
        }
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


    double forwardPass(double [] inputs,double output) {
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

        double [][] A = new double[linearParamCnt][1];

        // Linear part
        for (int i=0; i<defuzzVals.length; i++) {
            // bias parameter
            A[i*(inputs.length+1)][0] = normalizedVals[i];

            // mu*(input values)
            for (int j=0; j<inputs.length; j++) {
                A[i*(inputs.length+1)+1+j][0] = normalizedVals[i]*inputs[j];
            }
        }

        calculateLinear(A,output);
        linearParams = X[1];


        return -1;
    }

    // Calculate Error based on linear params and run backpropogation
    void backwardPass(double [] inputs, double output) {
        // Defuzzification
        for (int i=0; i<defuzzVals.length; i++) {
            // bias parameter
            defuzzVals[i] += linearParams[i*(inputs.length+1)] * normalizedVals[i];

            // linearParams * mu *(input values)
            for (int j=0; j<inputs.length; j++) {
                defuzzVals[i] += linearParams[i*(inputs.length+1)+1+j] * normalizedVals[i]*inputs[j];
            }

        }

        // Summations
        for (int i = 0; i<defuzzVals.length; i++) {
            outputVal += defuzzVals[i];
        }

        double error = Math.pow((output-outputVal),2);
    }


    // idx - index of the sample
    // A - n*(m+1)x1 sized vector
    // B - desired value
	void calculateLinear(double [][] A, double B) {
        double sum = 0.0;

		System.out.println("Size of S : ["+S.length+", "+S[0].length+"]");
		System.out.println("Size of A : ["+A.length+", "+A[0].length+"]");

        try {
			double [][] A_trans =  MatrixOperations.transpose(A);

			double [][] S_A 	= MatrixOperations.multiplySimple(S[0],A);
			double [][] AT_S 	= MatrixOperations.multiplySimple(A_trans,S[0]);
			double [][] AT_S_A = MatrixOperations.multiplySimple(A_trans,S_A);
			double [][] S_A_AT_S = MatrixOperations.multiplySimple(S_A,AT_S);

			/*
			System.out.println("Size of A_trans : ["+A_trans.length+", "+A_trans[0].length+"]");
			System.out.println("Size of S_A : ["+S_A.length+", "+S_A[0].length+"]");
			System.out.println("Size of AT_S : ["+AT_S.length+", "+AT_S[0].length+"]");
			System.out.println("Size of AT_S_A : ["+AT_S_A.length+", "+AT_S_A[0].length+"]");
			System.out.println("Size of S_A_AT_S : ["+S_A_AT_S.length+", "+S_A_AT_S[0].length+"]");
			*/

			// AT_S_A is 1x1 matrix
			double S_div = 1 + AT_S_A[0][0];
			double [][] S_frac = MatrixOperations.divideByNumber(S_A_AT_S,S_div);

			// Calculate next S[]
			S[1] = MatrixOperations.subtract(S[0],S_frac);


			// Calculating X....
            double [][] X_d = new double[1][X.length];
            X_d[0] = X[0];
            X_d = MatrixOperations.transpose(X_d);
            double [][] new_S_A = MatrixOperations.multiplySimple(S[1],A);
            double [][] AT_X = MatrixOperations.multiplySimple(A_trans,X_d);
            double diff = B - AT_X[0][0];
            System.out.println("Diff = "+diff);
            X[1] = MatrixOperations.add(X_d, MatrixOperations.multiplyByNumber(new_S_A,diff))[0];

            // keep current values as previous
            S[0] = S[1];
            X[0] = X[1];

		}
		catch (Exception ex) {
        	System.out.println(getClass().toString()+".calculateLinear(): "+ex);
		}

	}

	public double getOutputVal() {
		return outputVal;
	}
	
	public void runLearning() {
		int iterCnt = 0;
		double err = Double.MAX_VALUE;
		
		while ((err > minError) && (iterCnt++ < maxIterCnt) ) {
			//runForward();
			//calculateLinear();
			err = 0.0;
		}
	}
	
	public static void main (String [] args) {
		Anfis anfis = new Anfis();
		anfis.initialize();
		//anfis.runForward(new double[]{0.1, 0.2,-0.5,0.5});
		//System.out.println("Output: "+anfis.getOutputVal());

		double [][] A = new double[20][1];
		double [] B = new double[69];
		anfis.start(A,B);
		anfis.calculateLinear(A,B[0]);

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