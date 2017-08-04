package jml.anfis;

import jml.utils.FileOperations;
import jml.utils.MatrixOperations;

/**
 * Created by Jamal Hasanov on 6/26/2017.
 */
public class AnfisDemo {

    /**
     * Initialization of the ANFIS. This method initializes all layers (inputs, rules, etc) of the network.
     */
    public Anfis setParameters() {
        Anfis anfis = new Anfis(4,6);

        // setup activations
        anfis.activationList = new Activation[6];
        anfis.activationList[0] = new Activation(0, Activation.MembershipFunc.SIGMOID);
        anfis.activationList[1] = new Activation(0, Activation.MembershipFunc.BELL);
        anfis.activationList[2] = new Activation(1, Activation.MembershipFunc.SIGMOID);
        anfis.activationList[3] = new Activation(1, Activation.MembershipFunc.BELL);
        anfis.activationList[4] = new Activation(2, Activation.MembershipFunc.BELL);
        anfis.activationList[5] = new Activation(3, Activation.MembershipFunc.BELL);

        // setup rules
        anfis.ruleList = new Rule[4];
        anfis.ruleList[0] = new Rule((new int[]{0, 2, 4, 5}), Rule.RuleOperation.AND);
        anfis.ruleList[1] = new Rule((new int[]{1, 3, 4, 5}), Rule.RuleOperation.AND);
        anfis.ruleList[2] = new Rule((new int[]{1, 2, 4, 5}), Rule.RuleOperation.AND);
        anfis.ruleList[3] = new Rule((new int[]{0, 3, 4, 5}), Rule.RuleOperation.AND);

        // here anfis generates the remaining nodes
        anfis.init();

        return anfis;
    }

    /**
     * Train ANFIS network with parameters
     */
    public void trainAnfis() {
        //Anfis anfis = setParameters();
        Anfis anfis = Anfis.loadAnfisFromFile("ANFIS_conf_initial.xml");;

        double[][] A = FileOperations.readData("train_inputs.csv", ",");
        double[][] B = FileOperations.readData("train_outputs.csv", ",");
        //Convert [P][1] to [1][P] and then keep only first row (converting 2D array into 1D)
        try {
            B = MatrixOperations.transpose(B);
        } catch (Exception e) {
            e.printStackTrace();
        }

        int epochs = 100;
        double error = 0.01;
        System.out.println("Starting with:");
        System.out.println("epochs=" + epochs + "; error=" + error + " training data size=" + A.length + " ...");
        anfis.startHybridLearning(epochs, error, A, B[0], true, false);
        // Save ANFIS config in a file
        anfis.saveAnfisToFile("ANFIS_conf_trained.xml");
    }

    /**
     * Load ANFIS from config file and test given parameter
     */
    public void testAnfis() {
        //Anfis anfis = Anfis.loadAnfisFromFile("ANFIS_conf_trained.xml");;
        Anfis anfis = Anfis.loadAnfisFromFile("ANFIS_conf_matlab.xml");
        int totalCnt = 0;
        int truePos = 0;
        int falsePos = 0;
        int trueNeg = 0;
        int falseNeg = 0;

        /* When you need to test one value
        double[] retval = anfis.forwardPass(new double[] {0.3,0.3},-1, false);
        System.out.println("Retval="+retval[0]);
        */

        double [][] testData = FileOperations.readData("unit_test_data/test_ANFIS_2.csv",",");

        for (int i=0; i< testData.length; i++) {
            double [] inputs = new double[] {testData[i][0],testData[i][1]};
            double output = testData[i][2];
            double[] returnVal = anfis.forwardPass(inputs,-1, false);
            totalCnt++;

            if (returnVal[0] >= 0.9) {
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
        }

        System.out.println("TEST RESULT: ");
        System.out.println("Total samples: "+totalCnt);
        System.out.println("True Positives: "+truePos+"; True Negatives: "+trueNeg);
        System.out.println("False Positives: "+falsePos+"; False Negatives: "+falseNeg);
    }

    public static void main(String[] args) {
        /* Run this to train ANFIS network */
        new AnfisDemo().trainAnfis();

        /* Run this to test ANFIS network */
        //new AnfisDemo().testAnfis();
    }

}
