package jml.anfis.tests;

import jml.anfis.Activation;
import jml.anfis.Anfis;
import jml.utils.FileOperations;
import jml.utils.MatrixOperations;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
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

        for (int i=0; i<records.size(); i++) {
            ArrayList values = (ArrayList)records.get(i);
            x = (double)values.get(0);
            a = (double)values.get(1);
            b = (double)values.get(2);
            c = (double)values.get(3);
            expectation = (double)values.get(4);

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
        System.out.println("x="+x+"; a="+a+"; b="+b+"; c="+c);

        double funcVal_d1 = 0.0;
        double funcVal_d2 = 0.0;

        // Test A derivative
        funcVal_d1 = activationFunc.bellFunc(x, a+delta,b,c);
        funcVal_d2 = activationFunc.bellFunc(x, a-delta,b,c);
        double derivVal = activationFunc.bellFuncDerivA(x,a,b,c);
        double stepVal = (funcVal_d1 - funcVal_d2)/(2*delta);
        System.out.print("Derivative A... f'()="+derivVal+"; step val="+stepVal);
        assertTrue((derivVal - stepVal)<eps);
        System.out.println(" -> passed");

        // Test B derivative
        funcVal_d1 = activationFunc.bellFunc(x, a,b+delta,c);
        funcVal_d2 = activationFunc.bellFunc(x, a,b-delta,c);
        derivVal = activationFunc.bellFuncDerivB(x,a,b,c);
        stepVal = (funcVal_d1 - funcVal_d2)/(2*delta);
        System.out.print("Derivative B... f'()="+derivVal+"; step val="+stepVal);
        assertTrue(Math.abs(derivVal - stepVal)<eps);
        System.out.println(" -> passed");

        // Test C derivative
        funcVal_d1 = activationFunc.bellFunc(x, a,b,c+delta);
        funcVal_d2 = activationFunc.bellFunc(x, a,b,c-delta);
        derivVal = activationFunc.bellFuncDerivC(x,a,b,c);
        stepVal = (funcVal_d1 - funcVal_d2)/(2*delta);
        System.out.print("Derivative C... f'()="+derivVal+"; step val="+stepVal);
        assertTrue(Math.abs(derivVal - stepVal)<eps);
        System.out.println(" -> passed");
    }

    @Test
    public void testSigmoidDeriv() {
        double eps = 0.0001;
        double delta = 0.0000001;

        double x = Math.random();
        double a = Math.random();

        System.out.println("\n--- Testing Sigmoid function derivatives... ");
        System.out.println("x="+x+"; a="+a);

        double funcVal_d1 = activationFunc.sigmoidFunc(x, a+delta);
        double funcVal_d2 = activationFunc.sigmoidFunc(x, a-delta);
        double derivVal = activationFunc.sigmoidFuncDerivA(x,a);
        double stepVal = (funcVal_d1 - funcVal_d2)/(2*delta);
        System.out.print("Derivative of Sigmoid... f'()="+derivVal+"; step val="+stepVal);
        assertTrue(Math.abs(derivVal - stepVal)<eps);
        System.out.println(" -> passed");
    }

    @Test
    public void testNetworkGrad() {
        double eps = 0.0001;
        double delta = 0.0000001;

        double[][] A = FileOperations.readData("D:\\Aeroimages\\baku\\Jamal\\Codes\\Matlab\\PixelData\\anfis_brown_inputs.csv", ",");
        double[][] B = FileOperations.readData("D:\\Aeroimages\\baku\\Jamal\\Codes\\Matlab\\PixelData\\anfis_brown_outputs.csv", ",");
        //Convert [P][1] to [1][P] and then keep only first row (converting 2D array into 1D)
        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }

        double paramValue = 0.5;//Math.random();
        int exampleNo = (int)(A.length * Math.random());


        System.out.println("\n--- Testing whole network gradient calculation... ");
        System.out.println("param="+paramValue+"; exampleNo="+exampleNo);

        Anfis anfis = Anfis.loadAnfisFromFile("ANFIS_grad_test.xml");
        anfis.setActivationParamVal(0,0,paramValue+delta);
        System.out.println("param1="+anfis.getActivationParamVal(0,0));
        double[] res1 = anfis.forwardPass(A[exampleNo], -1,true);
        double func_val1 = Math.pow(B[0][exampleNo] - res1[0], 2) / 2;

        anfis.setActivationParamVal(0,0,paramValue-delta);
        System.out.println("param2="+anfis.getActivationParamVal(0,0));
        double[] res2 = anfis.forwardPass(A[exampleNo], -1,true);
        double func_val2 = Math.pow(B[0][exampleNo] - res2[0], 2) / 2;

        double stepVal = (func_val1-func_val2)/(2*delta);
        anfis.setActivationParamVal(0,0,paramValue);
        System.out.println("param3="+anfis.getActivationParamVal(0,0));
        Activation [] activationList = anfis.calculateGradient(A[exampleNo],B[0][exampleNo]);

        System.out.println("Activations paramDelta="+activationList[0].getParamDelta()[0]);
        System.out.println("Activations gradient="+activationList[0].getGradientVal());
        System.out.println("Step val="+stepVal);

        assertTrue(Math.abs(activationList[0].getParamDelta()[0] - stepVal)<eps);
        System.out.println(" -> passed");

    }


}