package jml.anfis.tests;

import jml.anfis.Activation;
import jml.anfis.Anfis;
import jml.utils.FileOperations;
import jml.utils.MatrixOperations;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;

import static org.junit.Assert.assertTrue;


/**
 * Created by itjamal on 6/8/2017.
 */
public class ActivationTest {
    Activation activationFunc;

    public ActivationTest() {
        activationFunc = new Activation();
    }

    ArrayList loadValidationData(String fileName) {
        ArrayList records = new ArrayList();

        try {
            File f = new File(fileName);
            FileReader fr = new FileReader(fileName);
            BufferedReader br = new BufferedReader(fr);

            String line = "";

            while ((line = br.readLine()) != null) {
                StringTokenizer tokenizer = new StringTokenizer(line, ",");
                ArrayList values = new ArrayList<Double>();
                while (tokenizer.hasMoreElements()) {
                    values.add(Double.parseDouble(tokenizer.nextToken()));
                }
                records.add(values);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return records;
    }

    @Test
    public void testTokenizer() {
        ArrayList records = loadValidationData("unit_test_data/TestValidation.txt");

        ArrayList values = (ArrayList) records.get(0);
        assertTrue(values.get(0).equals(Double.parseDouble("1")));
        assertTrue(values.get(1).equals(Double.parseDouble("0.1")));
        assertTrue(values.get(2).equals(Double.parseDouble("-1")));
        assertTrue(values.get(3).equals(Double.parseDouble("-0.1")));

        values = (ArrayList) records.get(1);
        assertTrue(values.get(0).equals(Double.parseDouble("0.222")));
        assertTrue(values.get(1).equals(Double.parseDouble("2.22")));
        assertTrue(values.get(2).equals(Double.parseDouble("22.2")));
        assertTrue(values.get(3).equals(Double.parseDouble("222")));
    }

    @Test
    public void bellFunc() {
        double eps = 0.001;
        double x, a, b, c, expectation;

        ArrayList records = loadValidationData("unit_test_data/BellFuncValidation.csv");

        for (int i = 0; i < records.size(); i++) {
            ArrayList values = (ArrayList) records.get(i);
            x = (double) values.get(0);
            a = (double) values.get(1);
            b = (double) values.get(2);
            c = (double) values.get(3);
            expectation = (double) values.get(4);

            assertTrue((expectation - activationFunc.bellFunc(x, a, b, c)) < eps);
        }
    }

    @Test
    public void testBellFuncDerivatives() {
        /*
        Testing derivatives by (f(x+d)-f(x))/d) rule
         */
        double eps = 0.0001;
        double delta = 0.0000001;

        double x = Math.random();
        double a = Math.random();
        double b = Math.random();
        double c = Math.random();

        System.out.println("\n--- Testing Bell function derivatives... ");
        System.out.println("x=" + x + "; a=" + a + "; b=" + b + "; c=" + c);

        double funcVal_d1 = 0.0;
        double funcVal_d2 = 0.0;

        // Test A derivative
        funcVal_d1 = activationFunc.bellFunc(x, a + delta, b, c);
        funcVal_d2 = activationFunc.bellFunc(x, a - delta, b, c);
        double derivVal = activationFunc.bellFuncDerivA(x, a, b, c);
        double stepVal = (funcVal_d1 - funcVal_d2) / (2 * delta);
        System.out.print("Derivative A... f'()=" + derivVal + "; step val=" + stepVal);
        assertTrue((derivVal - stepVal) < eps);
        System.out.println(" -> passed");

        // Test B derivative
        funcVal_d1 = activationFunc.bellFunc(x, a, b + delta, c);
        funcVal_d2 = activationFunc.bellFunc(x, a, b - delta, c);
        derivVal = activationFunc.bellFuncDerivB(x, a, b, c);
        stepVal = (funcVal_d1 - funcVal_d2) / (2 * delta);
        System.out.print("Derivative B... f'()=" + derivVal + "; step val=" + stepVal);
        assertTrue(Math.abs(derivVal - stepVal) < eps);
        System.out.println(" -> passed");

        // Test C derivative
        funcVal_d1 = activationFunc.bellFunc(x, a, b, c + delta);
        funcVal_d2 = activationFunc.bellFunc(x, a, b, c - delta);
        derivVal = activationFunc.bellFuncDerivC(x, a, b, c);
        stepVal = (funcVal_d1 - funcVal_d2) / (2 * delta);
        System.out.print("Derivative C... f'()=" + derivVal + "; step val=" + stepVal);
        assertTrue(Math.abs(derivVal - stepVal) < eps);
        System.out.println(" -> passed");
    }

    @Test
    public void testSigmoidDeriv() {
        double eps = 0.0001;
        double delta = 0.001;

        double x = Math.random();
        double a = Math.random();

        System.out.println("\n--- Testing Sigmoid function derivatives... ");
        System.out.println("x=" + x + "; a=" + a);

        double funcVal_d1 = activationFunc.sigmoidFunc(x, a + delta);
        double funcVal_d2 = activationFunc.sigmoidFunc(x, a - delta);
        double derivVal = activationFunc.sigmoidFuncDerivA(x, a);
        double stepVal = (funcVal_d1 - funcVal_d2) / (2 * delta);
        System.out.print("Derivative of Sigmoid... f'()=" + derivVal + "; step val=" + stepVal);
        assertTrue(Math.abs(derivVal - stepVal) < eps);
        System.out.println(" -> passed");
    }


    /**
     * Runs forward pass from given layer. It is used to debug the correctness of gradient calculation
     *
     * @param layerId     - the number of layer starting 1 as Activation Layer
     * @param inputs      - the inputs of the ANFIS. They are required in the layer 3
     * @param coefs       - the coefficients or consequent parameters
     * @param layerInputs - the inputs of the requested layer
     * @return
     */
    public double runFromGivenLayer(Anfis anfis, int layerId, double[] inputs, double[] coefs, double[] layerInputs) {
        double[] activations;
        double[] rule = new double[anfis.getRuleList().length];
        double[] norm = new double[anfis.getNormalizedVals().length];
        double[] funcVals = new double[norm.length];
        double[] defuzzVals = new double[norm.length];

        double output = 0.0;

        System.out.print("Layer values: ");
        for (int i = 0; i < layerInputs.length; i++)
            System.out.print(layerInputs[i] + " ");
        System.out.println();

        if (layerId == 1) {
            activations = layerInputs;

            rule[0] = activations[0] * activations[2] * activations[4];
            rule[1] = activations[1] * activations[3] * activations[5];
        } else if (layerId == 2) {
            rule = layerInputs;
        } else if (layerId == 3) {
            norm = layerInputs;
        } else if (layerId == 4)
            defuzzVals = layerInputs;

        if (layerId < 3) {
            double ruleSum = 0.0;
            for (int i = 0; i < rule.length; i++) {
                ruleSum += rule[i];
            }

            for (int i = 0; i < norm.length; i++) {
                norm[i] = rule[i] / ruleSum;
                System.out.println("norm[" + i + "]=" + norm[i]);
            }
        }
        if (layerId < 4) {
            for (int i = 0; i < norm.length; i++) {
                funcVals[i] = 0.0;
                for (int j = 0; j < inputs.length; j++) {
                    funcVals[i] += coefs[i * (inputs.length + 1) + j] * inputs[j];
                }
                // add bias parameter
                funcVals[i] += coefs[i * (inputs.length + 1) + inputs.length];

                defuzzVals[i] = funcVals[i] * norm[i];
            }
        }


        System.out.print("Defuzz: ");
        for (int i = 0; i < defuzzVals.length; i++) {
            System.out.print(" " + defuzzVals[i]);

            output += defuzzVals[i];
        }
        System.out.println("\nOutput: " + output);

        return output;
    }

    @Test
    public void testConsequtiveDerivatives() {
        double eps = 0.000001;
        double delta = 0.001;

        double[][] A = FileOperations.readData("E:\\Aeroimages\\Codes\\Matlab\\PixelData\\anfis_data_inputs.csv", ",");
        double[][] B = FileOperations.readData("E:\\Aeroimages\\Codes\\Matlab\\PixelData\\anfis_data_outputs.csv", ",");
        //Convert [P][1] to [1][P] and then keep only first row (converting 2D array into 1D)
        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_conf_trained.xml");
        System.out.println("\n--- Testing the derivative calculation of the CONSEQUITIVE parameters... ");

        int conseqIdx = 0;

        // Test all parameters
        while (conseqIdx < anfis.getLinearParamCnt()) {
            double[] prevLinearParams = anfis.getLinearParams();
            for (int exampleNo = 0; exampleNo < A.length; exampleNo++) {
                double[] consequentParams = Arrays.copyOf(prevLinearParams, prevLinearParams.length);
                consequentParams[conseqIdx] = prevLinearParams[conseqIdx] + eps * prevLinearParams[conseqIdx];
                anfis.setLinearParams(consequentParams);
                double[] res1 = anfis.forwardPass(A[exampleNo], -1, false);
                double func_val1 = Math.pow(B[0][exampleNo] - res1[0], 2) / 2.0;
//            System.out.println("param1=" + anfis.getActivationParamVal(inputIdx, paramIdx) + "; F1=" + func_val1);

                consequentParams[conseqIdx] = prevLinearParams[conseqIdx] - eps * prevLinearParams[conseqIdx];
                anfis.setLinearParams(consequentParams);
                double[] res2 = anfis.forwardPass(A[exampleNo], -1, false);
                double func_val2 = Math.pow(B[0][exampleNo] - res2[0], 2) / 2.0;
//            System.out.println("param2=" + anfis.getActivationParamVal(inputIdx, paramIdx) + "; F2=" + func_val2);

                double stepVal = (func_val1 - func_val2) / (2 * eps * prevLinearParams[conseqIdx]);

                anfis.setLinearParams(prevLinearParams);
                double[] res = anfis.forwardPass(A[exampleNo], -1, false);
                Activation[] activationList = anfis.calculateGradient(A[exampleNo], B[0][exampleNo], false);

                //System.out.println("Gradient = " + String.format("%.6f", anfis.getLinearParamGrads()[conseqIdx]) + "; stepVal=" + String.format("%.6f", stepVal));
                assertTrue(Math.abs(anfis.getLinearParamGrads()[conseqIdx] - stepVal) < delta);

                // Reset gradient values. Otherwise they'll be summed.
                anfis.resetGradientValues();
            }
            System.out.println(conseqIdx++ + " -> passed");
        }
    }

    @Test
    public void testPremiseDerivatives() {
        double eps = 0.000001;
        double delta = 0.001;

        double[][] A = FileOperations.readData("E:\\Aeroimages\\Codes\\Matlab\\PixelData\\anfis_data_inputs.csv", ",");
        double[][] B = FileOperations.readData("E:\\Aeroimages\\Codes\\Matlab\\PixelData\\anfis_data_outputs.csv", ",");
        //Convert [P][1] to [1][P] and then keep only first row (converting 2D array into 1D)
        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_conf_trained.xml");
        System.out.println("\n--- Testing whole network gradient calculation... ");

        int activationIdx = 0; // index of the input (membership function)

        Activation[] activations = anfis.getActivationList();
        while (activationIdx < activations.length) {
            int paramIdx = 0; // index of the activation's parameter
            while (paramIdx < activations[activationIdx].getParams().length) {
                double paramValue = anfis.getActivationParamVal(activationIdx, paramIdx);

                for (int exampleNo = 0; exampleNo < A.length; exampleNo++) {
                    anfis.setActivationParamVal(activationIdx, paramIdx, paramValue + eps * paramValue);
                    double[] res1 = anfis.forwardPass(A[exampleNo], -1, false);
                    double func_val1 = Math.pow(B[0][exampleNo] - res1[0], 2) / 2.0;
//            System.out.println("param1=" + anfis.getActivationParamVal(inputIdx, paramIdx) + "; F1=" + func_val1);

                    anfis.setActivationParamVal(activationIdx, paramIdx, paramValue - eps * paramValue);
                    double[] res2 = anfis.forwardPass(A[exampleNo], -1, false);
                    double func_val2 = Math.pow(B[0][exampleNo] - res2[0], 2) / 2.0;
//            System.out.println("param2=" + anfis.getActivationParamVal(inputIdx, paramIdx) + "; F2=" + func_val2);

                    double stepVal = (func_val1 - func_val2) / (2 * eps * paramValue);

                    anfis.setActivationParamVal(activationIdx, paramIdx, paramValue);
                    double[] res = anfis.forwardPass(A[exampleNo], -1, false);
                    Activation[] activationList = anfis.calculateGradient(A[exampleNo], B[0][exampleNo], false);

            /*
            if (activationList[inputIdx].getParamDelta()[paramIdx] != 0) {
                System.out.print("Inputs: ");
                for (int i = 0; i < A[exampleNo].length; i++) {
                    System.out.print("i[" + exampleNo + "][" + i + "]=" + A[exampleNo][i] + "; ");
                }
                System.out.print("Output= " + res[0] + "; desired=" + B[0][exampleNo] + "; ");
            }
            System.out.println("Gradient = " + String.format("%.6f",activationList[inputIdx].getParamDelta()[paramIdx]) + "; stepVal=" + String.format("%.6f",stepVal));
             */
            assertTrue(Math.abs(activationList[activationIdx].getParamDelta()[paramIdx] - stepVal) < delta);

                    // Reset gradient values. Otherwise they'll be summed.
                    anfis.resetGradientValues();
                }
                System.out.println("Activation: " + activationIdx + " Parameter: " + paramIdx++ + " -> passed");
            }
            activationIdx++;
        }
    }

            /*
            // This part is used to test the gradient of the special layer.
            //double[] layerValues = anfis.getActivationVals(); int startLayerId = 1;
            double[] layerValues = anfis.getRuleVals(); int startLayerId = 2;
            //double[] layerValues = anfis.getNormalizedVals(); int startLayerId = 3;

            double[] layerValues1 = new double[layerValues.length];
            double[] layerValues2 = new double[layerValues.length];;
            System.arraycopy(layerValues,0,layerValues1,0,layerValues.length);
            System.arraycopy(layerValues,0,layerValues2,0,layerValues.length);

            int targetElemIdx = 0; // index of the gradient element (or element in the layer)
            layerValues1[targetElemIdx] = layerValues[targetElemIdx] + eps;
            layerValues2[targetElemIdx] = layerValues[targetElemIdx] - eps;

            double output1 = runFromGivenLayer(anfis,startLayerId, A[exampleNo], anfis.getLinearParams(), layerValues1);
            func_val1 = Math.pow(B[0][exampleNo] - output1, 2) / 2.0;

            double output2 = runFromGivenLayer(anfis,startLayerId, A[exampleNo], anfis.getLinearParams(), layerValues2);
            func_val2 = Math.pow(B[0][exampleNo] - output2, 2) / 2.0;

            double s2 = (func_val1 - func_val2) / (2 * eps);
            System.out.println("Running from layer "+startLayerId+": func_val1="+func_val1+"; func_val2="+func_val2+"; s2 = " + String.format("%.6f",s2));
         */


}