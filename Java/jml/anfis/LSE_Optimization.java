package jml.anfis;

import jml.utils.MatrixOperations;

/**
 * Created by Jamal Hasanov on 6/18/2017.
 */
public class LSE_Optimization {
    double[][] X;
    double[][][] S;
    double gamma = 1000;


    public double[] findParameters(double [][] A, double [] B) {
        // initialize S matrix: S = gamma*I
        // gamma - positive large number
        // I     - Identity matrix
        // Note: we need to keep only 2 values of S and X - previous (index=0) and current (index=1)

        int paramCnt = A[0].length;
        S = new double[2][paramCnt][paramCnt];
        for (int i = 0; i < S[0].length; i++)
            S[0][i][i] = gamma * 1.0;

        // Used to find the best linear parameters
        X = new double[2][paramCnt];

        for (int recIdx = 0; recIdx < A.length; recIdx++) {
            // pass till normalization and keep results

            // This matrix stores input information for the LSE learning.
            // It stores the input of the defuzzification layer - output of the normalization layer and inputs to the ANFIS
            calculateLinearParams(A[recIdx], B[recIdx]);
            // keep current values as previous
            S[0] = S[1];
            X[0] = X[1];
        }

        return X[1];
    }

    /**
     * Calculates the Consequent Parameters using Least Squares Estimate (LSE)
     *
     * @param Inputs n*(m+1) sized vector
     * @param B desired value
     */
    public void calculateLinearParams(double[] Inputs, double B) {
        double sum = 0.0;

        try {
            // Doing this ugly job to make [N][1] array from [N] vector
            double [][] I = new double[1][Inputs.length];
            I[0] = Inputs;
            double [][] A = MatrixOperations.transpose(I);

            //System.out.println("Size of S : [" + S[0].length + ", " + S[0][0].length + "]");
            //System.out.println("Size of A : [" + A.length + ", " + A[0].length + "]");

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
        } catch (Exception ex) {
            System.out.println(getClass().toString() + ".calculateLinear(): " + ex);
        }
    }

}
