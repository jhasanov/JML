package jml.anfis;

import jml.utils.FileOperations;
import jml.utils.MatrixOperations;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class Anfis {
    double minError = 0.001;
    int maxIterCnt = 100;

    int inputCnt = 4;
    int activationCnt = 6;

    Activation[] activationList;
    Rule[] ruleList;

    double[] normalizedVals;
    double[] defuzzVals;
    double[] normalizedGrads;
    double[] defuzzGrads;
    double outputVal;

    double[] membershipParams;
    double[] linearParams;
    int linearParamCnt;

    double[][] X;
    double[][][] S;
    double gamma = 1000;

    public double getOutputVal() {
        return outputVal;
    }

    /**
     * Initialization of the ANFIS. This method initializes all layers (inputs, rules, etc) of the network.
     */
    public void initialize() {
        // setup activations
        activationList = new Activation[6];
        activationList[0] = new Activation(0, Activation.MembershipFunc.SIGMOID);
        activationList[1] = new Activation(0, Activation.MembershipFunc.BELL);
        activationList[2] = new Activation(1, Activation.MembershipFunc.SIGMOID);
        activationList[3] = new Activation(1, Activation.MembershipFunc.BELL);
        activationList[4] = new Activation(2, Activation.MembershipFunc.BELL);
        activationList[5] = new Activation(3, Activation.MembershipFunc.BELL);

        // setup rules
        ruleList = new Rule[4];
        ruleList[0] = new Rule((new int[]{0, 2, 3, 4}), Rule.RuleOperation.AND);
        ruleList[1] = new Rule((new int[]{1, 3, 3, 4}), Rule.RuleOperation.AND);
        ruleList[2] = new Rule((new int[]{1, 2, 3, 4}), Rule.RuleOperation.AND);
        ruleList[3] = new Rule((new int[]{0, 3, 3, 4}), Rule.RuleOperation.AND);

        normalizedVals = new double[ruleList.length];
        defuzzVals = new double[ruleList.length];

        // arrays to keep gradients of normalized and defuzzification layers
        normalizedGrads = new double[ruleList.length];
        defuzzGrads = new double[ruleList.length];

        // +1 bias parameter
        linearParamCnt = (inputCnt + 1) * ruleList.length;
        linearParams = new double[linearParamCnt];
        // Initialize linear parameters (coefficients) with random numbers
        for (int i = 0; i < linearParamCnt; i++)
            linearParams[i] = Math.random();

        for (int k = 0; k < activationCnt; k++) {
            if (activationList[k].mf == Activation.MembershipFunc.BELL) {
                System.out.println("   Initial Bell params: ("+activationList[k].params[0]+","+activationList[k].params[1]+","+activationList[k].params[2]+")");
            } else if (activationList[k].mf == Activation.MembershipFunc.SIGMOID) {
                System.out.println("Initial Sigmoid params: ("+activationList[k].params[0]+")");
            }
        }
    }

    /**
     * Performs forward pass to get the output value. Just pass input values and get output.
     *
     * @param inputs      values for the ANFIS input
     * @param tillLayerID layer till which forward pass shall go (inclusively).
     *                    If tillLayerID = 3, the pass will go through Activation (1), Rules (2) and Normalization(3) layers.
     *                    If tillLayerID = -1, pass goes till the end and output is returned as double[1]
     * @return output of the ANFIS layer
     */
    double[] forwardPass(double[] inputs, int tillLayerID) {
        double[] layerOutput;

        // calculate Activation values
        layerOutput = new double[activationCnt];
        for (int i = 0; i < activationList.length; i++) {
            activationList[i].activate(inputs);
            layerOutput[i] = activationList[i].activationVal;
        }

        if (tillLayerID == 1)
            return layerOutput;


        // calculate Rules and total sum of them (for normalization phase)
        layerOutput = new double[ruleList.length];
        double ruleSum = 0.0;
        for (int i = 0; i < ruleList.length; i++) {
            ruleSum += ruleList[i].calculate(activationList);
            layerOutput[i] = ruleList[i].getRuleVal();
        }

        if (tillLayerID == 2)
            return layerOutput;

        // Normalize - iterate through rules and calculate normalized value
        for (int i = 0; i < normalizedVals.length; i++) {
            normalizedVals[i] = ruleList[i].getRuleVal() / ruleSum;
        }

        if (tillLayerID == 3)
            return normalizedVals;

        // Defuzzification
        // 	 Od = normVals * (k0 + k1*x1 + k2*x2 + ... + kn*xn)
        // where
        //   Od 		- Output of defuzzification layer
        //   normVals 	- Input of the defuzzification layer ( output of the corresponding normalization node)
        //   xi 		- ith input of the ANFIS
        //   ki         - linear parameters, where k0 is bias
        for (int i = 0; i < defuzzVals.length; i++) {
            for (int j = 0; j < inputs.length; j++) {
                defuzzVals[i] += linearParams[i * (inputs.length + 1) + j] * inputs[j];
            }
            // add bias parameter
            defuzzVals[i] += linearParams[i * (inputs.length + 1) + inputs.length];
            defuzzVals[i] *= normalizedVals[i];
        }

        if (tillLayerID == 4)
            return defuzzVals;

        // Summations
        layerOutput = new double[1];
        for (int i = 0; i < defuzzVals.length; i++) {
            layerOutput[0] += defuzzVals[i];
        }
        return layerOutput;
    }

    /**
     * Hybrid Learning algorithm
     *
     * @param epochCnt count of maximum epochs (iterations over training set)
     * @param inputs   all training inputs
     * @param outputs  desired outputs
     */
    void startHybridLearning(int epochCnt, double minError, double[][] inputs, double[] outputs) {
        // learning rate
        double alpha = 0.01; // use formula

        // sum of all errors
        double totalError = Double.MAX_VALUE;
        // iteration count
        int iterCnt = 0;

        // repeat until error is minimized or max epoch count is reached
        while ((totalError > minError) && (iterCnt++ < epochCnt)) {
            // ---- First iterate over all input data and find Consequent Parameters

            // initialize S matrix: S = gamma*I
            // gamma - positive large number
            // I     - Identity matrix
            // Note: we need to keep only 2 values of S - previous (index=0) and current (index=1)
            S = new double[2][linearParamCnt][linearParamCnt];
            for (int i = 0; i < S.length; i++)
                S[0][i][i] = gamma * 1.0;

            // Used to find the best linear parameters
            X = new double[2][linearParamCnt];


            // iterate over the training set - batch mode
            for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
                // pass till normalization and keep results
                double[] x = inputs[recIdx];
                double[] normOutput = forwardPass(x, 3);

                //System.out.print("Iteration: " + recIdx);

                // This matrix stores input information for the LSE learning.
                // It stores the input of the defuzzification layer - output of the normalization layer and inputs to the ANFIS
                double[][] A = new double[linearParamCnt][1];

                for (int j = 0; j < defuzzVals.length; j++) {
                    // bias parameter
                    A[j * (x.length + 1)][0] = normOutput[j];

                    // input values
                    for (int k = 0; k < x.length; k++) {
                        A[j * (x.length + 1) + 1 + k][0] = normOutput[j] * x[k];
                    }
                }
                calculateLinearParams(A, outputs[recIdx]);
            }

            System.out.println("Consequent Parameters found");
            // update consequent params
            linearParams = X[1];

            // --- Iterate over all input data and find Premise Parameters
            totalError = 0.0;
            System.out.println("Backpropogation starts");
            for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
                // pass till the end and calculate output value
                double[] outputValue = forwardPass(inputs[recIdx], -1);

                // calculate error
                totalError += Math.pow(outputValue[0] - outputs[recIdx], 2);
                // --- Run BACK PROPOGATION ---

                // calculate Error->Output->Defuzz->Normalization gradients
                for (int k = 0; k < defuzzVals.length; k++) {

                    double f = 0.0;
                    for (int m = 0; m < inputs[recIdx].length; m++) {
                        f += linearParams[k * (inputs[recIdx].length + 1) + m] * inputs[recIdx][m];
                    }
                    // add bias parameter
                    f += linearParams[k * (inputs[recIdx].length + 1) + inputs[recIdx].length];

                    normalizedGrads[k] = alpha * (outputValue[0] - outputs[recIdx]) * f * normalizedVals[k] * (1 - normalizedVals[k]);
                }

                // calculate Normalization->Rules gradients
                for (int k = 0; k < ruleList.length; k++) {
                    // Iterate over each "Rule<->Normalization" connection
                    double sum = 0.0;
                    ruleList[k].gradientVal = 0.0;
                    for (int m = 0; m < normalizedGrads.length; m++) {
                        ruleList[k].gradientVal += normalizedGrads[m] * (1 / ruleList[k].getRuleVal());
                    }
                }

                // calculate gradients for each membership function (Rules->Membership)
                // first reset all values
                for (int k = 0; k < activationCnt; k++) {
                    activationList[k].gradientVal = 0.0;
                }

                for (int k = 0; k < ruleList.length; k++) {
                    // find all connected membership functions to each rule and summarize the value
                    for (int m = 0; m < ruleList[k].inputActivations.length; m++) {
                        int idx = ruleList[k].inputActivations[m];

                        activationList[idx].gradientVal += ruleList[k].ruleVal;
                    }
                }

                // Now, find final gradients!
                for (int k = 0; k < activationCnt; k++) {
                    if (activationList[k].mf == Activation.MembershipFunc.BELL) {
                        // derivatives A,B and C
                        double ad, bd, cd;

                        ad = activationList[k].calcDerivative(inputs[recIdx], 1);
                        bd = activationList[k].calcDerivative(inputs[recIdx], 2);
                        cd = activationList[k].calcDerivative(inputs[recIdx], 3);

                        activationList[k].params_delta[0] += ad * activationList[k].gradientVal;
                        activationList[k].params_delta[1] += bd * activationList[k].gradientVal;
                        activationList[k].params_delta[2] += cd * activationList[k].gradientVal;
                    } else if (activationList[k].mf == Activation.MembershipFunc.SIGMOID) {
                        // derivatives A
                        double ad;

                        ad = activationList[k].calcDerivative(inputs[recIdx], 1);

                        activationList[k].params_delta[0] += ad * activationList[k].gradientVal;
                    }
                }
            }

            // adjust parameters
            for (int k = 0; k < activationCnt; k++) {
                if (activationList[k].mf == Activation.MembershipFunc.BELL) {
                    activationList[k].params[0] += activationList[k].params_delta[0]/inputs.length;
                    activationList[k].params[1] += activationList[k].params_delta[1]/inputs.length;
                    activationList[k].params[2] += activationList[k].params_delta[2]/inputs.length;
                    System.out.println("   Bell params: ("+activationList[k].params[0]+","+activationList[k].params[1]+","+activationList[k].params[2]+")");
                } else if (activationList[k].mf == Activation.MembershipFunc.SIGMOID) {
                    activationList[k].params[0] += activationList[k].params_delta[0]/inputs.length;
                    System.out.println("Sigmoid params: ("+activationList[k].params[0]+")");
                }
            }

            System.out.println("Epoch = " + iterCnt + " Total Error = " + totalError);
        }

    }

    /**
     * Calculates the Consequent Parameters using Least Squares Estimate (LSE)
     *
     * @param A n*(m+1)x1 sized vector
     * @param B desired value
     */
    void calculateLinearParams(double[][] A, double B) {
        double sum = 0.0;

        /*
        System.out.println("Size of S : [" + S[0].length + ", " + S[0][0].length + "]");
        System.out.println("Size of A : [" + A.length + ", " + A[0].length + "]");
        */

        try {
            double[][] A_trans = MatrixOperations.transpose(A);

            double[][] S_A = MatrixOperations.multiplySimple(S[0], A);
            double[][] AT_S = MatrixOperations.multiplySimple(A_trans, S[0]);
            double[][] AT_S_A = MatrixOperations.multiplySimple(A_trans, S_A);
            double[][] S_A_AT_S = MatrixOperations.multiplySimple(S_A, AT_S);

            /*
            System.out.println("Size of A_trans : ["+A_trans.length+", "+A_trans[0].length+"]");
			System.out.println("Size of S_A : ["+S_A.length+", "+S_A[0].length+"]");
			System.out.println("Size of AT_S : ["+AT_S.length+", "+AT_S[0].length+"]");
			System.out.println("Size of AT_S_A : ["+AT_S_A.length+", "+AT_S_A[0].length+"]");
			System.out.println("Size of S_A_AT_S : ["+S_A_AT_S.length+", "+S_A_AT_S[0].length+"]");
            */

            // AT_S_A is 1x1 matrix
            double S_div = 1 + AT_S_A[0][0];
            double[][] S_frac = MatrixOperations.divideByNumber(S_A_AT_S, S_div);

            // Calculate next S[]
            S[1] = MatrixOperations.subtract(S[0], S_frac);

            // Calculating X....
            double[][] X_d = new double[1][X.length];
            X_d[0] = X[0];
            X_d = MatrixOperations.transpose(X_d);
            double[][] new_S_A = MatrixOperations.multiplySimple(S[1], A);
            double[][] AT_X = MatrixOperations.multiplySimple(A_trans, X_d);
            double diff = B - AT_X[0][0];
            //System.out.println("; Diff = " + diff);
            double[][] res = MatrixOperations.add(X_d, MatrixOperations.multiplyByNumber(new_S_A, diff));
            X[1] = MatrixOperations.transpose(res)[0];

            // keep current values as previous
            S[0] = S[1];
            X[0] = X[1];
        } catch (Exception ex) {
            System.out.println(getClass().toString() + ".calculateLinear(): " + ex);
        }
    }


    public static void main(String[] args) {
        Anfis anfis = new Anfis();
        anfis.initialize();

        double[][] A = FileOperations.readData("D:\\Dropbox\\Public\\inputs.csv", ",");
        double[][] B = FileOperations.readData("D:\\Dropbox\\Public\\outputs.csv", ",");
        //Convert [P][1] to [1][P] and then keep only first row (converting 2D array into 1D)
        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }

        int epochs = 10;
        double error = 0.001;
        System.out.println("Starting with:");
        System.out.println("epochs=" + epochs + "; error=" + error + " training data size=" + A.length + " ...");
        anfis.startHybridLearning(epochs, error, A, B[0]);
    }
}

class Input {
    Activation activation;
}

class Rule {
    enum RuleOperation {AND, OR}

    ;

    RuleOperation oper;
    int[] inputActivations;
    double ruleVal;
    // the value of the calculated gradient till this node
    double gradientVal = 0.0;

    public Rule(int[] inputMFidx, RuleOperation oper) {
        this.inputActivations = inputMFidx;
        this.oper = oper;
    }

    public double calculate(Activation[] mfInputs) {
        if (oper == RuleOperation.AND) {
            ruleVal = 1.0;

            for (int i = 0; i < inputActivations.length; i++) {
                ruleVal = ruleVal * mfInputs[inputActivations[i]].getActivationVal();
            }
        } else if (oper == RuleOperation.OR) {
            ruleVal = 1.0;

            for (int i = 0; i < inputActivations.length; i++) {
                ruleVal = Math.min(ruleVal, mfInputs[inputActivations[i]].getActivationVal());
            }
        }

        return ruleVal;
    }

    public double getRuleVal() {
        return ruleVal;
    }

}