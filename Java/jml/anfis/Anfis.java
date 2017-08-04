package jml.anfis;

import com.sun.xml.internal.txw2.output.IndentingXMLStreamWriter;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.swing.*;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;

public class Anfis {
    int inputCnt = 0;
    int activationCnt = 0;

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

    public Anfis(int inputCnt, int activationCnt) {
        this.inputCnt = inputCnt;
        this.activationCnt = activationCnt;
    }


    /**
     * Creates all layer connection and set initial values.
     */
    public void init() {
        normalizedVals = new double[ruleList.length];
        defuzzVals = new double[ruleList.length];

        // arrays to keep gradients of normalized and defuzzification layers
        normalizedGrads = new double[ruleList.length];
        defuzzGrads = new double[ruleList.length];

        // +1 bias parameter
        linearParamCnt = (inputCnt + 1) * ruleList.length;
        // if not given in XML file
        if ((linearParams == null) || (linearParams.length == 0)) {
            linearParams = new double[linearParamCnt];
            // Initialize linear parameters (coefficients) with random numbers
            for (int i = 0; i < linearParamCnt; i++)
                linearParams[i] = Math.random();
        }

        for (int k = 0; k < activationCnt; k++) {
            // if no params given in XML
            if (activationList[k].params == null)
                activationList[k].setRandomParams();
        }

        printActivationParams("Initial");
    }

    // Print activation function parameters:
    public void printActivationParams(String prefix) {
        System.out.println("____________________________________________");
        for (int k = 0; k < activationCnt; k++) {
            if (activationList[k].mf == Activation.MembershipFunc.BELL) {
                System.out.println(prefix + " Bell params: (" + activationList[k].params[0] + "," + activationList[k].params[1] + "," + activationList[k].params[2] + ")");
            } else if (activationList[k].mf == Activation.MembershipFunc.SIGMOID) {
                System.out.println(prefix + " Sigmoid params: (" + activationList[k].params[0] + ")");
            }
        }
    }

    /**
     * Saves ANFIS structure and parameters in a file
     *
     * @param filename XML file to store the ANFIS data
     */
    public void saveAnfisToFile(String filename) {
        try {
            System.out.println("Saving ANFIS to file: " + filename);
            OutputStream outputStream = new FileOutputStream(new File(filename));
            XMLOutputFactory xmlOF = XMLOutputFactory.newInstance();
            // This line of code makes XML structure formatted
            XMLStreamWriter xmlWrite = new IndentingXMLStreamWriter(xmlOF.createXMLStreamWriter(outputStream));

            xmlWrite.writeStartDocument();
            xmlWrite.writeStartElement("anfis");

            // Storing ANFIS structure
            xmlWrite.writeStartElement("structure");
            xmlWrite.writeAttribute("inputs", "" + inputCnt);
            xmlWrite.writeAttribute("activations", "" + activationCnt);
            xmlWrite.writeAttribute("rules", "" + ruleList.length);
            xmlWrite.writeEndElement();

            // Writing Activation Layer
            xmlWrite.writeStartElement("layer");
            xmlWrite.writeAttribute("id", "1");
            xmlWrite.writeAttribute("desc", "ACTIVATION");

            for (int i = 0; i < activationCnt; i++) {
                xmlWrite.writeStartElement("param");
                xmlWrite.writeAttribute("id", "" + (i + 1));
                xmlWrite.writeAttribute("MF", activationList[i].mf.toString());

                // add parameters
                xmlWrite.writeStartElement("input");
                xmlWrite.writeAttribute("id", "" + (activationList[i].inputNo + 1));
                xmlWrite.writeEndElement();

                // add coefficients
                for (int j = 0; j < activationList[i].params.length; j++) {
                    xmlWrite.writeStartElement("coef");
                    xmlWrite.writeAttribute("id", "" + (j + 1));
                    xmlWrite.writeAttribute("val", "" + activationList[i].params[j]);
                    xmlWrite.writeEndElement();
                }

                xmlWrite.writeEndElement();
            }
            xmlWrite.writeEndElement();

            // Writing Rule Layer
            xmlWrite.writeStartElement("layer");
            xmlWrite.writeAttribute("id", "2");
            xmlWrite.writeAttribute("desc", "RULE");

            for (int i = 0; i < ruleList.length; i++) {
                xmlWrite.writeStartElement("param");
                xmlWrite.writeAttribute("id", "" + (i + 1));
                xmlWrite.writeAttribute("OPERATION", ruleList[i].oper.toString());

                // add inputs
                for (int j = 0; j < ruleList[i].inputActivations.length; j++) {
                    xmlWrite.writeStartElement("input");
                    xmlWrite.writeAttribute("id", "" + (ruleList[i].inputActivations[j] + 1));
                    xmlWrite.writeEndElement();
                }

                xmlWrite.writeEndElement();
            }

            xmlWrite.writeEndElement();

            // Writing Linear (Consequent) Parameters
            xmlWrite.writeStartElement("consequent_coefs");
            xmlWrite.writeAttribute("count", "" + linearParamCnt);

            for (int i = 0; i < linearParamCnt; i++) {
                xmlWrite.writeStartElement("consequent_coef");
                xmlWrite.writeAttribute("idx", "" + (i + 1));
                xmlWrite.writeAttribute("val", "" + linearParams[i]);
                xmlWrite.writeEndElement();
            }

            xmlWrite.writeEndElement();

            // close document tag
            xmlWrite.writeEndDocument();
            xmlWrite.close();
        } catch (Exception ex) {
            System.out.println("Anfis.saveAnfisToFile(): " + filename);
        }

    }

    /**
     * Load Anfis structure and parameters from file (SAP parser is used).
     *
     * @param filename XML file with the ANFIS structure
     */
    public static Anfis loadAnfisFromFile(String filename) {
        Anfis anfis = null;
        try {
            System.out.println("Loading ANFIS from file: " + filename);
            File inputFile = new File(filename);
            SAXParserFactory factory = SAXParserFactory.newInstance();
            SAXParser saxParser = factory.newSAXParser();
            AnfisXmlHandler anfisXml = new AnfisXmlHandler();
            saxParser.parse(inputFile, anfisXml);

            anfis = new Anfis(anfisXml.inputCnt, anfisXml.activationCnt);
            anfis.activationList = anfisXml.getActivationList();
            anfis.ruleList = anfisXml.getRuleList();
            anfis.linearParams = anfisXml.getLinearParams();
            anfis.init();
        } catch (Exception ex) {
            System.out.println("Anfis.loadAnfisFromFile(): " + ex);
        } finally {
            return anfis;
        }

    }

    /**
     * Performs forward pass to get the output value. Just pass input values and get output.
     *
     * @param inputs      values for the ANFIS input
     * @param tillLayerID layer till which forward pass shall go (inclusively).
     *                    If tillLayerID = 3, the pass will go through Activation (1), Rules (2) and Normalization(3) layers.
     *                    If tillLayerID = -1, pass goes till the end and output is returned as double[1]
     * @param bVerbose    if TRUE, shows the values of all layers
     * @return output of the ANFIS layer
     */
    double[] forwardPass(double[] inputs, int tillLayerID, boolean bVerbose) {
        double[] layerOutput;

        if (bVerbose)
            System.out.println("Layer 1 outputs:");
        // calculate Activation values
        layerOutput = new double[activationCnt];
        for (int i = 0; i < activationList.length; i++) {
            activationList[i].activate(inputs);
            layerOutput[i] = activationList[i].activationVal;
            if (bVerbose)
                System.out.print("" + layerOutput[i] + " ");
        }
        if (bVerbose)
            System.out.println();

        if (tillLayerID == 1)
            return layerOutput;

        if (bVerbose)
            System.out.println("Layer 2 outputs:");

        // calculate Rules and total sum of them (for normalization phase)
        layerOutput = new double[ruleList.length];
        double ruleSum = 0.0;
        for (int i = 0; i < ruleList.length; i++) {
            ruleSum += ruleList[i].calculate(activationList);
            layerOutput[i] = ruleList[i].getRuleVal();
            if (bVerbose)
                System.out.print("" + layerOutput[i] + " ");
        }

        if (bVerbose)
            System.out.println();

        if (tillLayerID == 2)
            return layerOutput;

        if (bVerbose)
            System.out.println("Layer 3 outputs:");

        // Normalize - iterate through rules and calculate normalized value
        for (int i = 0; i < normalizedVals.length; i++) {
            normalizedVals[i] = ruleList[i].getRuleVal() / ruleSum;
            if (bVerbose)
                System.out.print("" + normalizedVals[i] + " ");
        }

        if (bVerbose)
            System.out.println();

        if (tillLayerID == 3)
            return normalizedVals;

        if (bVerbose)
            System.out.println("Layer 4 outputs:");

        // Defuzzification
        // 	 Od = normVals * (k0 + k1*x1 + k2*x2 + ... + kn*xn)
        // where
        //   Od 		- Output of defuzzification layer
        //   normVals 	- Input of the defuzzification layer ( output of the corresponding normalization node)
        //   xi 		- ith input of the ANFIS
        //   ki         - linear parameters, where k0 is bias

        // First reset values
        for (int i = 0; i < defuzzVals.length; i++) {
            defuzzVals[i] = 0.0;
        }

        for (int i = 0; i < defuzzVals.length; i++) {
            for (int j = 0; j < inputs.length; j++) {
                defuzzVals[i] += linearParams[i * (inputs.length + 1) + j] * inputs[j];
            }
            // add bias parameter
            defuzzVals[i] += linearParams[i * (inputs.length + 1) + inputs.length];
            defuzzVals[i] *= normalizedVals[i];
            if (bVerbose)
                System.out.print("" + defuzzVals[i] + " ");
        }

        if (bVerbose)
            System.out.println();

        if (tillLayerID == 4)
            return defuzzVals;

        // Summations
        if (bVerbose)
            System.out.println("Layer 5 outputs:");

        layerOutput = new double[1];
        for (int i = 0; i < defuzzVals.length; i++) {
            layerOutput[0] += defuzzVals[i];
        }
        if (bVerbose)
            System.out.println("" + layerOutput[0]);
        return layerOutput;
    }

    /**
     * Hybrid Learning algorithm
     *
     * @param epochCnt count of maximum epochs (iterations over training set)
     * @param inputs   all training inputs
     * @param outputs  desired outputs
     */
    void startHybridLearning(int epochCnt, double minError, double[][] inputs, double[] outputs, boolean bVisualize, boolean bDebug) {
        // variables of the "Golden section search" method
        double a = 0.0;
        double b = 10.0;
        double[] alpha = new double[2];
        double[] Err_GSS = new double[2];
        double[][][] weights = new double[2][activationCnt][3];
        double eps = 0.00000001;

        // used to reset parameters when value is NaN
        boolean bReset = false;
        double[] errors = new double[epochCnt];
        double maxError = 0.0;
        GraphPanel graphPanel = new GraphPanel();

        // save old activation values for visualization
        Activation[] oldActivations = activationList.clone();
        for (int i = 0; i < activationCnt; i++)
            oldActivations[i] = activationList[i].clone();

        if (bVisualize) {
            JFrame frame = new JFrame("ANFIS Learning");
            frame.setSize(600, 300);
            frame.add(graphPanel);
            frame.setVisible(true);
        }

        // iteration count
        int iterCnt = 0;
        // set to maximum to satisfy the (errors[0] > minError) check below (in "while")
        errors[0] = Double.MAX_VALUE;

        // measure initial Error
        double initError = 0.0;
        for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
            // pass till the end and calculate output value
            double[] outputValue = forwardPass(inputs[recIdx], -1, false);
            initError += (outputs[recIdx] - outputValue[0]);
        }
        System.out.println("Error before training: "+initError);


            // repeat until a)error is minimized, b)max epoch count is reached or c) alpha range is too small
        while ((errors[iterCnt++] > minError) && (iterCnt < epochCnt) && (Math.abs(b - a) > eps)) {
            // This matrix stores input information for the LSE learning.
            // It stores the input of the defuzzification layer - output of the normalization layer and inputs to the ANFIS
            double[][] A = new double[inputs.length][linearParamCnt];
            int recIdx;

            for (recIdx = 0; recIdx < inputs.length; recIdx++) {
                double[] x = inputs[recIdx];

                // pass till normalization and keep results
                double[] normOutput = forwardPass(x, 3, false);

                for (int j = 0; j < defuzzVals.length; j++) {
                    // input values
                    for (int k = 0; k < x.length; k++) {
                        A[recIdx][j * (x.length + 1) + k] = normOutput[j] * x[k];
                    }
                    // bias parameter
                    A[recIdx][j * (x.length + 1) + x.length] = normOutput[j];
                }
            }

            // Runs Sequental LSE in batch mode to find consequent parameters
            LSE_Optimization lse = new LSE_Optimization();
            double[] linearP = lse.findParameters(A, outputs);
            linearParams = linearP;

            // --- Iterate over all input data and find Premise Parameters

            // update variables for "Golden Section Search (GSS)"
            alpha[0] = b - (b - a) / 1.618;
            alpha[1] = a + (b - a) / 1.618;

            // --- Run BACK PROPOGATION ---

            if (bDebug)
                printActivationParams("New epoch");

            // ------------------------------------
            // one-dimentional minimization: calculate Error function for 2 values:
            weights = new double[2][activationCnt][3];
            for (int aidx = 0; aidx < alpha.length; aidx++) {
                // Save current activation parameters for the next option as well
                // Both functions shall be calculated with the same initial parameters
                if (aidx == 0) {
                    for (int k = 0; k < activationCnt; k++) {
                        for (int n = 0; n < activationList[k].params.length; n++)
                            activationList[k].params_prev[n] = activationList[k].params[n];
                    }
                }

                if (bDebug) {
                    // print delimeter
                    System.out.println("____________________________________________");
                }
                // Iterate through all training samples
                for (recIdx = 0; recIdx < inputs.length; recIdx++) {
                    // pass till the end and calculate output value
                    double[] outputValue = forwardPass(inputs[recIdx], -1, false);
                    double diff = (outputs[recIdx] - outputValue[0]);

                    /* Used for debug : to see what are the outputs for the first 10 samples
                    if (recIdx < 10) {
                        System.out.println("recIdx=" + recIdx + "; diff = " + diff);
                        if (Double.isNaN(diff)) {
                            System.out.println("Diff is NaN because (" + outputs[recIdx] + "-" + outputValue + ")=NaN");
                        }
                    }
                    */

                    // calculate Error->Output->Defuzz->Normalization gradients
                    for (int k = 0; k < defuzzVals.length; k++) {
                        double f = 0.0;
                        for (int m = 0; m < inputs[recIdx].length; m++) {
                            f += linearParams[k * (inputs[recIdx].length + 1) + m] * inputs[recIdx][m];
                        }
                        // add bias parameter
                        f += linearParams[k * (inputs[recIdx].length + 1) + inputs[recIdx].length];
                        normalizedGrads[k] = alpha[aidx] * diff * f * normalizedVals[k] * (1 - normalizedVals[k]);
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

                            activationList[idx].gradientVal += ruleList[k].gradientVal + 0.001; // add 0.001 to avoid zero
                        }
                    }

                    // Now, find final gradients!
                    for (int k = 0; k < activationCnt; k++) {
                        if (bReset) {
                            // if there's a need to reset, do not find final gradients, stop the for loop
                            // and start with the first sample (recIdx = 0).
                            bReset = false;
                            recIdx = -1;
                            break;
                        }

                        if (activationList[k].mf == Activation.MembershipFunc.BELL) {
                            // derivatives A,B and C
                            double ad, bd, cd;

                            ad = activationList[k].calcDerivative(inputs[recIdx], 1);
                            bd = activationList[k].calcDerivative(inputs[recIdx], 2);
                            cd = activationList[k].calcDerivative(inputs[recIdx], 3);

                            if ((Double.isNaN(ad)) || (Double.isNaN(bd)) || (Double.isNaN(cd))) {
                                System.out.println("NaN - Bell - resetting");
                                for (int l = 0; l < activationCnt; l++)
                                    activationList[l].setRandomParams();
                                printActivationParams("Reset");
                                bReset = true;
                                continue;
                            } else {
                                activationList[k].params_delta[0] += ad * activationList[k].gradientVal;
                                activationList[k].params_delta[1] += bd * activationList[k].gradientVal;
                                activationList[k].params_delta[2] += cd * activationList[k].gradientVal;
                            }
                        } else if (activationList[k].mf == Activation.MembershipFunc.SIGMOID) {
                            // derivatives A
                            double ad;

                            ad = activationList[k].calcDerivative(inputs[recIdx], 1);
                            if ((Double.isNaN(ad))) {
                                System.out.println("NaN - Sigm - resetting.");
                                for (int l = 0; l < activationCnt; l++)
                                    activationList[l].setRandomParams();
                                printActivationParams("Reset");
                                bReset = true;
                                continue;
                            } else
                                activationList[k].params_delta[0] += ad * activationList[k].gradientVal;
                        }
                    }
                }

                // Check if all inputs were trained (otherwise, RESET was requested)
                if (recIdx == inputs.length) {
                    // adjust parameters
                    for (int k = 0; k < activationCnt; k++) {
                        if (activationList[k].mf == Activation.MembershipFunc.BELL) {
                            activationList[k].params[0] += activationList[k].params_delta[0] / inputs.length;
                            activationList[k].params[1] += activationList[k].params_delta[1] / inputs.length;
                            activationList[k].params[2] += activationList[k].params_delta[2] / inputs.length;
                            weights[aidx][k][0] += (activationList[k]).params[0];
                            weights[aidx][k][1] += (activationList[k]).params[1];
                            weights[aidx][k][2] += (activationList[k]).params[2];
                            if (bDebug) {
                                System.out.println("New weight for Bell. [" + aidx + "," + k + ",0]; weight=" + weights[aidx][k][0]);
                                System.out.println("New weight for Bell. [" + aidx + "," + k + ",1]; weight=" + weights[aidx][k][1]);
                                System.out.println("New weight for Bell. [" + aidx + "," + k + ",2]; weight=" + weights[aidx][k][2]);
                            }
                            // resetting the weight adjustment
                            activationList[k].params_delta[0] = 0.0;
                            activationList[k].params_delta[1] = 0.0;
                            activationList[k].params_delta[2] = 0.0;
                        } else if (activationList[k].mf == Activation.MembershipFunc.SIGMOID) {
                            activationList[k].params[0] += activationList[k].params_delta[0] / inputs.length;
                            weights[aidx][k][0] += (activationList[k]).params[0];
                            if (bDebug) {
                                System.out.println("New weight for Sigmoid. [" + aidx + "," + k + ",0]; weight=" + weights[aidx][k][0]);
                            }
                            // resetting the weight adjustment
                            activationList[k].params_delta[0] = 0.0;
                        }
                    }
                }

                // Calculate error for given parameters
                Err_GSS[aidx] = 0;
                for (recIdx = 0; recIdx < inputs.length; recIdx++) {
                    double[] outputValue = forwardPass(inputs[recIdx], -1, false);
                    Err_GSS[aidx] += Math.pow(outputs[recIdx] - outputValue[0], 2) / 2;
                }

                // Reset activation params for the next option (when first part is done)
                if (aidx == 0) {
                    for (int k = 0; k < activationCnt; k++) {
                        for (int n = 0; n < activationList[k].params.length; n++)
                            activationList[k].params[n] = activationList[k].params_prev[n];
                    }
                }
            }

            int winnerIdx = 1; // option that gave minimum error
            if (Err_GSS[0] > Err_GSS[1]) {
                a = alpha[0];
            } else {
                b = alpha[1];
                winnerIdx = 0;
            }

            // Setting weight to the optimal options (with smaller Error)
            for (int k = 0; k < activationCnt; k++) {
                for (int n = 0; n < activationList[k].params.length; n++) {
                    activationList[k].params[n] = weights[winnerIdx][k][n];
                }
            }

            errors[iterCnt - 1] = Math.min(Err_GSS[0], Err_GSS[1]);
            maxError = Math.max(maxError, errors[iterCnt - 1]);

            // will be used in WHILE check
            if (iterCnt < epochCnt)
                errors[iterCnt] = Double.MAX_VALUE;

            graphPanel.setData(maxError, errors);
            System.out.println("____________________________________________");
            System.out.println("Epoch = " + iterCnt + "; alpha in ( " + alpha[0] + "," + alpha[1] + "); Error = " + errors[iterCnt - 1] + "; (Err0=" + Err_GSS[0] + "; Err1=" + Err_GSS[1]+")");
        }

        // Print parameters of Membership Functions after learning
        printActivationParams("Final");

        // Visualize Membership functions (show initial and updated MFs)
        if (bVisualize) {
            JFrame[] mfFrame = new JFrame[activationCnt];
            int frameWidth = 400;
            int framwHeight = 300;
            int horizWndCnt = 3; // count of windows in one horizontal line
            int pad = 20; // space between adjacent windows
            for (int k = 0; k < activationCnt; k++) {
                // Draw graph in [-10,10] range (to see how it looks like) but outline behaviour in our [-1,1] range
                MFGraph mfg = new MFGraph(activationList[k], oldActivations[k], -3, 3, -1, 1);
                mfFrame[k] = new JFrame("Activation " + k);
                mfFrame[k].setSize(frameWidth, framwHeight);
                mfFrame[k].setLocation((k % horizWndCnt) * frameWidth + pad, (k / horizWndCnt) * framwHeight + pad);
                mfFrame[k].add(mfg);
                mfFrame[k].setVisible(true);
            }
        }
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

class AnfisXmlHandler extends DefaultHandler {
    ArrayList<Activation> activationList = new ArrayList<Activation>();
    ArrayList<Rule> ruleList = new ArrayList<Rule>();
    int[] ruleArr;

    Activation activation;
    double[] linearParams;

    int inputCnt = 0;
    int activationCnt = 0;
    int ruleCnt = 0;
    int layerId = 1;
    int activationId = 0;
    int ruleIdx = 0;
    String activationMF = "";
    String ruleOperation = "";

    @Override
    public void startElement(String uri,
                             String localName, String qName, Attributes attributes)
            throws SAXException {
        if (qName.equalsIgnoreCase("structure")) {
            inputCnt = Integer.parseInt(attributes.getValue("inputs"));
            activationCnt = Integer.parseInt(attributes.getValue("activations"));
            ruleCnt = Integer.parseInt(attributes.getValue("rules"));
            ruleArr = new int[ruleCnt];
            System.out.print("Input count: " + inputCnt + ", ");
            System.out.print("Activation count: " + activationCnt + ", ");
            System.out.println("Rule count: " + ruleCnt);
        } else if (qName.equalsIgnoreCase("layer")) {
            layerId = Integer.parseInt(attributes.getValue("id"));
            System.out.println("Layer ID : " + layerId);
        } else if (qName.equalsIgnoreCase("param")) {
            activationId = Integer.parseInt(attributes.getValue("id"));
            activationMF = attributes.getValue("MF");
            ruleOperation = attributes.getValue("OPERATION");
            System.out.println("   Params : (" + activationId + "," + activationMF + "," + ruleOperation + ")");
        } else if (qName.equalsIgnoreCase("input")) {
            int inputId = Integer.parseInt(attributes.getValue("id"));
            System.out.println("      Input: " + inputId);
            // In case of Activation Layer
            if (layerId == 1) {
                activation = new Activation(inputId - 1, Activation.MembershipFunc.valueOf(activationMF));
            } else if (layerId == 2) {
                ruleArr[ruleIdx++] = inputId - 1; // because of Java array index
            }
        } else if (qName.equalsIgnoreCase("coef")) {
            int coefId = Integer.parseInt(attributes.getValue("id"));
            String val = attributes.getValue("val");
            if (val == null)
                activation.params[coefId - 1] = Math.random();
            else
                activation.params[coefId - 1] = Double.parseDouble(val);
            System.out.println("      Coef: (" + coefId + "," + activation.params[coefId - 1] + ")");
        } else if (qName.equalsIgnoreCase("consequent_coefs")) {
            int paramCnt = Integer.parseInt(attributes.getValue("count"));
            linearParams = new double[paramCnt];
        } else if (qName.equalsIgnoreCase("consequent_coef")) {
            int paramIdx = Integer.parseInt(attributes.getValue("idx"));
            String val = attributes.getValue("val");
            if (val == null)
                linearParams[paramIdx - 1] = Math.random();
            else
                linearParams[paramIdx - 1] = Double.parseDouble(val);
            System.out.println("      Consequent Coef: (" + paramIdx + "," + linearParams[paramIdx - 1] + ")");
        }
    }

    @Override
    public void endElement(String uri,
                           String localName, String qName) throws SAXException {
        if (qName.equalsIgnoreCase("layer")) {
            System.out.println("/Layer");
        } else if (qName.equalsIgnoreCase("param")) {
            if (layerId == 1) {
                // add Activation object to a list
                activationList.add(activation);
            }
            if (layerId == 2) {
                // add Rule object to a list
                Rule rule = new Rule(ruleArr, Rule.RuleOperation.valueOf(ruleOperation));
                ruleList.add(rule);
                ruleIdx = 0;
                ruleArr = new int[ruleCnt];
            }
        }
    }

    public int getInputCnt() {
        return inputCnt;
    }

    public int getActivationCnt() {
        return activationCnt;
    }

    public double[] getLinearParams() {
        return linearParams;
    }

    public Activation[] getActivationList() {
        return activationList.toArray(new Activation[activationList.size()]);
    }

    public Rule[] getRuleList() {
        return ruleList.toArray(new Rule[ruleList.size()]);
    }

}