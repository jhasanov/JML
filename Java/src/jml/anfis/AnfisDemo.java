package jml.anfis;

import jml.utils.FileOperations;
import jml.utils.MatrixOperations;

import javax.swing.*;

/**
 * Created by Jamal Hasanov on 6/26/2017.
 */
public class AnfisDemo {

    public Anfis configCustomAnfis() {
        Anfis anfis = new Anfis(6, 18);
        anfis.activationList = new Activation[18];
        anfis.activationList[0] = new Activation(0, Activation.MembershipFunc.SIGMOID,new double[]{-0.5});
        anfis.activationList[1] = new Activation(0, Activation.MembershipFunc.SIGMOID,new double[]{0.5});
        anfis.activationList[2] = new Activation(0, Activation.MembershipFunc.SIGMOID,null);
        anfis.activationList[3] = new Activation(1, Activation.MembershipFunc.SIGMOID,new double[]{0.5});
        anfis.activationList[4] = new Activation(1, Activation.MembershipFunc.SIGMOID,new double[]{0.5});
        anfis.activationList[5] = new Activation(1, Activation.MembershipFunc.SIGMOID,null);
        anfis.activationList[6] = new Activation(2, Activation.MembershipFunc.SIGMOID,new double[]{0.5});
        anfis.activationList[7] = new Activation(2, Activation.MembershipFunc.SIGMOID,new double[]{-0.5});
        anfis.activationList[8] = new Activation(2, Activation.MembershipFunc.SIGMOID,null);

        anfis.activationList[9] = new Activation(3, Activation.MembershipFunc.CENTERED_BELL,null);
        anfis.activationList[10] = new Activation(3, Activation.MembershipFunc.CENTERED_BELL,null);
        anfis.activationList[11] = new Activation(3, Activation.MembershipFunc.CENTERED_BELL,null);
        anfis.activationList[12] = new Activation(4, Activation.MembershipFunc.CENTERED_BELL,null);
        anfis.activationList[13] = new Activation(4, Activation.MembershipFunc.CENTERED_BELL,null);
        anfis.activationList[14] = new Activation(4, Activation.MembershipFunc.CENTERED_BELL,null);
        anfis.activationList[15] = new Activation(5, Activation.MembershipFunc.CENTERED_BELL,null);
        anfis.activationList[16] = new Activation(5, Activation.MembershipFunc.CENTERED_BELL,null);
        anfis.activationList[17] = new Activation(5, Activation.MembershipFunc.CENTERED_BELL,null);

        // setup rules
        anfis.ruleList = new Rule[14];
        anfis.ruleList[0] = new Rule((new int[]{0,3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[1] = new Rule((new int[]{1,4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[2] = new Rule((new int[]{2,5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[3] = new Rule((new int[]{0,1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[4] = new Rule((new int[]{3,4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[5] = new Rule((new int[]{6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[6] = new Rule((new int[]{0, 1, 2, 12, 13, 14, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[7] = new Rule((new int[]{3, 4, 5, 9, 10, 11, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[8] = new Rule((new int[]{6, 7, 8, 9, 10, 11, 12, 13, 14}), Rule.RuleOperation.AND);
        anfis.ruleList[9] = new Rule((new int[]{0, 1, 2, 9, 10 ,11 }), Rule.RuleOperation.AND);
        anfis.ruleList[10] = new Rule((new int[]{3, 4, 5, 12, 13, 14,}), Rule.RuleOperation.AND);
        anfis.ruleList[11] = new Rule((new int[]{6, 7, 8, 15, 16, 17}), Rule.RuleOperation.AND);
        anfis.ruleList[12] = new Rule((new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8 }), Rule.RuleOperation.AND);
        anfis.ruleList[13] = new Rule((new int[]{9, 10, 11, 12, 13, 14, 15, 16, 17}), Rule.RuleOperation.AND);

        // here anfis generates the remaining nodes
        anfis.init();

        return anfis;
    }

    /**
     * Initialization of the ANFIS. This method initializes all layers (inputs, rules, etc) of the network.
     */
    public Anfis setParameters() {
        // setup activations
        byte INPUTS = 1;
        byte MF_PER_INPUT = 8;

        Anfis anfis = new Anfis(INPUTS, INPUTS * MF_PER_INPUT);

        anfis.activationList = new Activation[INPUTS * MF_PER_INPUT];
        for (int inputIdx = 0; inputIdx < INPUTS; inputIdx++) {
            for (int mfIdx = 0; mfIdx < MF_PER_INPUT; mfIdx++) {
                if ((mfIdx % 2) == 0)
                    anfis.activationList[inputIdx * MF_PER_INPUT + mfIdx] = new Activation(inputIdx, Activation.MembershipFunc.SIGMOID,null);
                else
                    anfis.activationList[inputIdx * MF_PER_INPUT + mfIdx] = new Activation(inputIdx, Activation.MembershipFunc.BELL,null);
            }
        }


        // setup rules
        // auto-generate the combination of the rules:
        int ruleCount = anfis.activationCnt * (anfis.activationCnt - 1) / 2;
        anfis.ruleList = new Rule[ruleCount];
        int rIdx = 0;
        for (int inputIdx = 0; inputIdx < anfis.activationCnt; inputIdx++) {
            for (int otherInputIdx = 0; otherInputIdx < anfis.activationCnt; otherInputIdx++) {
                if ((otherInputIdx != inputIdx) && (otherInputIdx > inputIdx))
                    anfis.ruleList[rIdx++] = new Rule((new int[]{inputIdx, otherInputIdx}), Rule.RuleOperation.AND);
            }
        }

        // here anfis generates the remaining nodes
        anfis.init();

        return anfis;
    }

    /**
     * Train ANFIS network with parameters
     */
    public void trainAnfis() {
        boolean bDither = false;
        //Anfis anfis = setParameters();
        Anfis anfis = configCustomAnfis();
        FileOperations.saveAnfisToFile(anfis,"ANFIS_config_v2.xml");
//        Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_config_sincos.xml");
//        Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_config_v2.xml");

        //double[][] A = FileOperations.readData("generated_input.csv", ",");
        //double[][] B = FileOperations.readData("generated_output.csv", ",");
//        double[][] A = FileOperations.readData("../../ColorCloseness/Matlab/sample_data/cos_sin_func.csv", ",",0,1,false);
//        double[][] B = FileOperations.readData("../../ColorCloseness/Matlab/sample_data/cos_sin_func.csv", ",",1,2,false);
        double[][] A = FileOperations.readData("../../ColorCloseness/python/trainingData.csv", ",",0,6,true);
        double[][] B = FileOperations.readData("../../ColorCloseness/python/trainingData.csv", ",",6,7,true);

        // Dithering parameters by %5
        if (bDither) {
            for (int i = 0; i < A.length; i++)
                for (int j = 0; j < A[i].length; j++) {
                    if (Math.random() > 0.5)
                        A[i][j] = A[i][j] * (1.1);
                    else
                        A[i][j] = A[i][j] * (0.9);

                }
        }

        //Convert [P][1] to [1][P] and then keep only first row (converting 2D array into 1D)
        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }

        int epochs = 150;
        double error = 0.0001;
        System.out.println("Starting with:");
        System.out.println("epochs=" + epochs + "; error=" + error + " training data size=" + A.length + " ...");
        //anfis.startTraining(false, epochs, error, A, B[0], true, false);
        anfis.adamLearning(20,epochs, error, A, B[0], true, false);
        // Save ANFIS config in a file
        FileOperations.saveAnfisToFile(anfis, "ANFIS_conf_trained_adam.xml");
    }

    /**
     * Load ANFIS from config file and test given parameter
     */
    public void testAnfis(double premiseParamDev, double conseqParamDev, double inputDev) {
        Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_conf_trained_bp.xml");
        int totalCnt = 0;
        int truePos = 0;
        int falsePos = 0;
        int trueNeg = 0;
        int falseNeg = 0;

        // Update premise parameters by given deviation
        if (premiseParamDev != 0) {
            System.out.println("Updating the premise parameter by " + premiseParamDev);
            Activation[] activations = anfis.getActivationList();
            for (int j = 0; j < activations.length; j++) {
                double[] apar = activations[j].getParams();
                for (int i = 0; i < apar.length; i++) {
                    anfis.setActivationParamVal(j, i, apar[i] * (1 + premiseParamDev));
                }
            }
        }
        // Update consequent parameters by given deviation
        if (conseqParamDev != 0) {
            System.out.println("Updating the consequent parameter by " + conseqParamDev);
            double[] linear = anfis.getLinearParams();

            for (int i = 0; i < linear.length; i++) {
                linear[i] = linear[i] * (1 + conseqParamDev);
            }
            anfis.setLinearParams(linear);
        }

        // When you need to test one value
        //double[] retval = anfis.forwardPass(new double[] {6*1.0/180, 106*1.0/255, 12*1.0/255},-1, true);
        //System.out.println("Retval="+retval[0]);

        double[][] A = FileOperations.readData("../../ColorCloseness/python/testData.csv", ",",0,6,true);
        double[][] B = FileOperations.readData("../../ColorCloseness/python/testData.csv", ",",6,7,true);
        //double[][] A = FileOperations.readData("generated_input.csv", ",");
        //double[][] B = FileOperations.readData("generated_output.csv", ",");
        if (inputDev != 0) {
            System.out.println("Updating the inputs by " + inputDev);
            for (int i = 0; i < A.length; i++)
                for (int j = 0; j < A[i].length; j++)
                    A[i][j] = A[i][j] * (1 + inputDev);
        }

        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }


        double minOut = Double.MAX_VALUE;
        double maxOut = Double.MIN_VALUE;

        // calculate M.S.E. and output range
        double mse = 0.0;
        for (int i = 0; i < A.length; i++) {
            double[] returnVal = anfis.forwardPass(A[i], -1, false);
            mse += Math.pow(returnVal[0] - B[0][i], 2.0) / 2;

            minOut = Math.min(minOut, returnVal[0]);
            maxOut = Math.max(maxOut, returnVal[0]);
        }

        int res001 = 0;
        int res002 = 0;
        int res003 = 0;

        for (int i = 0; i < A.length; i++) {
            double[] returnVal = anfis.forwardPass(A[i], -1, false);
            totalCnt++;

            if (Math.abs(returnVal[0] - B[0][i]) < 0.01 * (maxOut - minOut))
                res001++;
            else if (Math.abs(returnVal[0] - B[0][i]) < 0.02 * (maxOut - minOut))
                res002++;
            else if (Math.abs(returnVal[0] - B[0][i]) < 0.05 * (maxOut - minOut))
                res003++;

            /*if (returnVal[0] >= 0.9) {
                if (output == 1)
                    truePos++;
                else
                    falsePos++;
            }
            else if (returnVal[0] < 0.9) {
                if (output == 0)
                    trueNeg++;
                else
                    falseNeg++;
            }

             */
        }

        System.out.println("Min: " + minOut + "; Max: " + maxOut);
        System.out.println("TEST RESULT: ");
        System.out.println("Total samples: " + totalCnt);
        System.out.println("M.S.E. : " + mse / totalCnt);
        System.out.println("Positives: 1%: " + res001 + "; 2%: " + res002 + "; 3%: " + res003);
        //System.out.println("True Positives: "+truePos+"; True Negatives: "+trueNeg);
        //System.out.println("False Positives: "+falsePos+"; False Negatives: "+falseNeg);
    }

    public void testIteratedAnfis() {
        Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_conf_trained_sincos_hybrid.xml");
        int totalCnt = 0;

        // calculate M.S.E. and output range
        double mse = 0.0;
        for (double x = -2; x <=2; x+=0.001) {
            double[] returnVal = anfis.forwardPass(new double[]{x}, -1, false);
            //double y = Math.cos(x*x)*Math.sin(x*x*x);
            System.out.println(x+","+returnVal[0]);
        }

    }

    public void generateAnfisData(String anfisConfigFile, String outputFile, int sampleCount) {
        Anfis anfis = FileOperations.loadAnfisFromFile(anfisConfigFile);

        double[] inputs;

        for (int i = 0; i < sampleCount; i++) {
            inputs = new double[anfis.getInputCnt()];
            String textLine = "";
            for (int j = 0; j < inputs.length; j++) {
                inputs[j] = Math.random();
                textLine += inputs[j];
                if (j == (inputs.length - 1))
                    textLine += "\n";
                else
                    textLine += ",";
            }
            FileOperations.appendToFile(outputFile + "_input.csv", textLine);
            double[] returnVal = anfis.forwardPass(inputs, -1, false);
            textLine = returnVal[0] + "\n";
            FileOperations.appendToFile(outputFile + "_output.csv", textLine);
        }
    }

    public static void main(String[] args) {
        /* Run this to train ANFIS network */
        new AnfisDemo().trainAnfis();

        /* Run this to test ANFIS network */
        //new AnfisDemo().testAnfis(0, 0, 0);
        //new AnfisDemo().testIteratedAnfis();

        /* Generate data with ANFIS */
        //Anfis anfis = new AnfisDemo().setParameters();
        //FileOperations.saveAnfisToFile(anfis,"ANFIS_original.xml");
        //new AnfisDemo().generateAnfisData("ANFIS_original.xml","generated",1000);
    }

}
