package jml.anfis;

import jml.utils.FileOperations;
import jml.utils.MatrixOperations;

import javax.swing.*;

/**
 * Created by Jamal Hasanov on 6/26/2017.
 */
public class AnfisDemo {

    /**
     * Initialization of the ANFIS. This method initializes all layers (inputs, rules, etc) of the network.
     */
    public Anfis setParameters() {
        // setup activations
        byte INPUTS = 2;
        byte MF_PER_INPUT = 2;

        Anfis anfis = new Anfis(INPUTS,INPUTS*MF_PER_INPUT);

        anfis.activationList = new Activation[INPUTS* MF_PER_INPUT];
        for (int inputIdx = 0; inputIdx<INPUTS; inputIdx++) {
            for (int mfIdx = 0; mfIdx <  MF_PER_INPUT; mfIdx++) {
                if ((mfIdx % 2) == 0)
                    anfis.activationList[inputIdx* MF_PER_INPUT+mfIdx] = new Activation(inputIdx, Activation.MembershipFunc.BELL);
                else
                    anfis.activationList[inputIdx* MF_PER_INPUT+mfIdx] = new Activation(inputIdx, Activation.MembershipFunc.BELL);
            }

        }

        // setup rules
        // auto-generate the combination of the rules:
        int ruleCount = anfis.activationCnt * (anfis.activationCnt-1)/2;
        anfis.ruleList = new Rule[ruleCount];
        int rIdx = 0;
        for (int inputIdx = 0; inputIdx < anfis.activationCnt; inputIdx++) {
            for (int otherInputIdx = 0; otherInputIdx < anfis.activationCnt; otherInputIdx++) {
                if ((otherInputIdx != inputIdx) && (otherInputIdx> inputIdx))
                    anfis.ruleList[rIdx++] = new Rule((new int[]{inputIdx,otherInputIdx}), Rule.RuleOperation.AND);
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
        Anfis anfis = setParameters();
        //FileOperations.saveAnfisToFile(anfis,"ANFIS_initial.xml");
        //Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_start.xml");

        //double[][] A = FileOperations.readData("generated_input.csv", ",");
        //double[][] B = FileOperations.readData("generated_output.csv", ",");
        double[][] A = FileOperations.readData("celab2000_sample_input-1k.csv", ",");
        double[][] B = FileOperations.readData("celab2000_sample_output-1k.csv", ",");

        //Convert [P][1] to [1][P] and then keep only first row (converting 2D array into 1D)
        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }

        int epochs = 100;
        double error = 0.0000001;
        System.out.println("Starting with:");
        System.out.println("epochs=" + epochs + "; error=" + error + " training data size=" + A.length + " ...");
        anfis.startTraining(false,epochs, error, A, B[0], true, false);
        // Save ANFIS config in a file
        FileOperations.saveAnfisToFile(anfis,"ANFIS_conf_trained.xml");
    }

    /**
     * Load ANFIS from config file and test given parameter
     */
    public void testAnfis(double premiseParamDev,double conseqParamDev,double inputDev) {
        Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_original.xml");
        //Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_start.xml");
        //Anfis anfis = FileOperations.loadAnfisFromFile("ANFIS_conf_trained.xml");
        int totalCnt = 0;
        int truePos = 0;
        int falsePos = 0;
        int trueNeg = 0;
        int falseNeg = 0;

        // Update premise parameters by given deviation
        if (premiseParamDev != 0) {
            System.out.println("Updating the premise parameter by "+premiseParamDev);
            Activation[] activations = anfis.getActivationList();
            for (int j = 0; j <activations.length; j++) {
                double [] apar = activations[j].getParams();
                for (int i =0;i <apar.length;i++) {
                    anfis.setActivationParamVal(j,i,apar[i]*(1+premiseParamDev));
                }
            }
        }
        // Update consequent parameters by given deviation
        if (conseqParamDev != 0) {
            System.out.println("Updating the consequent parameter by "+conseqParamDev);
            double [] linear = anfis.getLinearParams();

            for (int i =0;i <linear.length;i++) {
                linear[i] = linear[i]*(1+conseqParamDev);
            }
            anfis.setLinearParams(linear);
        }

        // When you need to test one value
        //double[] retval = anfis.forwardPass(new double[] {6*1.0/180, 106*1.0/255, 12*1.0/255},-1, true);
        //System.out.println("Retval="+retval[0]);

        //double[][] A = FileOperations.readData("celab2000_sample_input-1k.csv", ",");
        //double[][] B = FileOperations.readData("celab2000_sample_output-1k.csv", ",");
        double[][] A = FileOperations.readData("generated_input.csv", ",");
        double[][] B = FileOperations.readData("generated_output.csv", ",");

        if (inputDev != 0) {
            System.out.println("Updating the inputs by "+inputDev);
            for (int i =0; i<A.length; i++)
                for (int j=0; j<A[i].length;j++)
                    A[i][j] = A[i][j]*(1+inputDev);
        }

        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }


        double minOut = Double.MAX_VALUE;
        double maxOut = Double.MIN_VALUE;

        for (int i=0; i< A.length; i++) {
            double[] returnVal = anfis.forwardPass(A[i],-1, false);
            totalCnt++;

            minOut = Math.min(minOut,returnVal[0]);
            maxOut = Math.max(maxOut,returnVal[0]);
            if (Math.abs(returnVal[0] - B[0][i]) < 0.01)
                truePos++;
            else
                trueNeg++;

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

        System.out.println("Min: "+minOut+ "; Max: "+maxOut);
        System.out.println("TEST RESULT: ");
        System.out.println("Total samples: "+totalCnt);
        System.out.println("True Positives: "+truePos+"; True Negatives: "+trueNeg);
        System.out.println("False Positives: "+falsePos+"; False Negatives: "+falseNeg);
    }

    public void generateAnfisData(String anfisConfigFile, String outputFile,int sampleCount) {
        Anfis anfis = FileOperations.loadAnfisFromFile(anfisConfigFile);

        double [] inputs;

        for(int i = 0; i< sampleCount; i++) {
            inputs = new double[anfis.getInputCnt()];
            String textLine = "";
            for (int j = 0; j<inputs.length; j++){
                inputs[j] = Math.random();
                textLine += inputs[j];
                if (j == (inputs.length -1))
                    textLine += "\n";
                else
                    textLine += ",";
            }
            FileOperations.appendToFile(outputFile+"_input.csv",textLine);
            double[] returnVal = anfis.forwardPass(inputs,-1, false);
            textLine = returnVal[0] + "\n";
            FileOperations.appendToFile(outputFile+"_output.csv",textLine);
        }
    }

    public static void main(String[] args) {
        /* Run this to train ANFIS network */
        //new AnfisDemo().trainAnfis();

        /* Run this to test ANFIS network */
        new AnfisDemo().testAnfis(0,0,0.1);

        /* Generate data with ANFIS */
        //Anfis anfis = new AnfisDemo().setParameters();
        //FileOperations.saveAnfisToFile(anfis,"ANFIS_conf_original.xml");
        //new AnfisDemo().generateAnfisData("ANFIS_original.xml","generated",1000);
    }

}
