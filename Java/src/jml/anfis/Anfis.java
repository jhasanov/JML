package jml.anfis;

import javax.swing.*;

public class Anfis {
    int inputCnt = 0;
    int activationCnt = 0;

    Activation[] activationList;
    Rule[] ruleList;

    double[] ruleVals;
    double[] normalizedVals;
    double[] funcVals;
    double[] defuzzVals;
    double[] normalizedGrads;
    double[] defuzzGrads;

    double[] linearParams;
    double[] linearParamsPrev;
    double[] linearParamGrads;
    int linearParamCnt;
    int premiseParamCnt;
    int consequtiveParamCnt;

    public Anfis(int inputCnt, int activationCnt) {
        this.inputCnt = inputCnt;
        this.activationCnt = activationCnt;
    }

    public int getInputCnt() {
        return inputCnt;
    }

    public void setInputCnt(int inputCnt) {
        this.inputCnt = inputCnt;
    }

    public int getActivationCnt() {
        return activationCnt;
    }

    public void setActivationCnt(int activationCnt) {
        this.activationCnt = activationCnt;
    }

    public Rule[] getRuleList() {
        return ruleList;
    }

    public void setRuleList(Rule[] ruleList) {
        this.ruleList = ruleList;
    }

    public Activation[] getActivationList() {
        return activationList;
    }

    public void setActivationList(Activation[] activationList) {
        this.activationList = activationList;
    }

    public double[] getLinearParams() {
        return linearParams;
    }

    public void setLinearParams(double[] linearParams) {
        this.linearParams = linearParams;
        this.linearParamsPrev = new double[linearParams.length];
        this.linearParamGrads = new double[linearParams.length];
    }

    public int getLinearParamCnt() {
        return linearParamCnt;
    }

    public void setLinearParamCnt(int linearParamCnt) {
        this.linearParamCnt = linearParamCnt;
    }

    public double[] getLinearParamGrads() {
        return linearParamGrads;
    }

    /**
     * Creates all layer connection and set initial values.
     */
    public void init() {
        normalizedVals = new double[ruleList.length];
        funcVals = new double[ruleList.length];
        defuzzVals = new double[ruleList.length];

        // arrays to keep gradients of normalized and defuzzification layers
        normalizedGrads = new double[ruleList.length];
        defuzzGrads = new double[ruleList.length];

        // +1 bias parameter
        linearParamCnt = (inputCnt + 1) * ruleList.length;
        //linearParamCnt = (activationCnt + 1) * ruleList.length; //MF to Defuzz

        // if not given in XML file
        if ((linearParams == null) || (linearParams.length == 0)) {
            linearParams = new double[linearParamCnt];
            linearParamsPrev = new double[linearParamCnt];

            linearParamGrads = new double[linearParamCnt];
            // Initialize linear parameters (coefficients) with random numbers
            for (int i = 0; i < linearParamCnt; i++)
                linearParams[i] = Math.random();///linearParamCnt; // make linear params smaller to decrease their impact.
        }

        for (int k = 0; k < activationCnt; k++) {
            // if no params given in XML
            if (activationList[k].params == null)
                activationList[k].setRandomParams();
        }

        resetGradientValues();
        printActivationParams("Initial");
    }

    /**
     * Used to change the value of the parameter. Example: you want to set 2'nd input's MF (Bell function) 2nd parameter to 0.3:
     * setActivationParamVal(1,1,0.3)
     * Indexing starts from 0.
     *
     * @param mfIndex    index of the input MF
     * @param paramIndex index of the MF's parameter
     * @param value      value of the parameter
     */
    public void setActivationParamVal(int mfIndex, int paramIndex, double value) {
        activationList[mfIndex].params[paramIndex] = value;
    }

    public double getActivationParamVal(int mfIndex, int paramIndex) {
        return activationList[mfIndex].params[paramIndex];
    }

    public double[] getActivationVals() {
        double[] retval = new double[activationList.length];

        for (int i = 0; i < activationList.length; i++) {
            retval[i] = activationList[i].activationVal;
        }

        return retval;
    }

    public double[] getRuleVals() {
        return ruleVals;
    }

    public double[] getNormalizedVals() {
        return normalizedVals;
    }

    public double[] getDefuzzVals() {
        return defuzzVals;
    }

    // Print activation function parameters:
    public void printActivationParams(String prefix) {
        System.out.println("____________________________________________");
        for (int k = 0; k < activationCnt; k++) {
            if (activationList[k].mf == Activation.MembershipFunc.BELL) {
                System.out.println(prefix + " Bell params: (" + activationList[k].params[0] + "," + activationList[k].params[1] + "," + activationList[k].params[2] + ")");
                //System.out.println("Prev Bell params: (" + activationList[k].params_prev[0] + "," + activationList[k].params_prev[1] + "," + activationList[k].params_prev[2] + ")");
            }
            else if (activationList[k].mf == Activation.MembershipFunc.CENTERED_BELL) {
                    System.out.println(prefix + " Centered Bell params: (" + activationList[k].params[0] + "," + activationList[k].params[1] + ")");
                    //System.out.println("Prev Bell params: (" + activationList[k].params_prev[0] + "," + activationList[k].params_prev[1] + "," + activationList[k].params_prev[2] + ")");
            } else if (activationList[k].mf == Activation.MembershipFunc.SIGMOID) {
                System.out.println(prefix + " Sigmoid params: (" + activationList[k].params[0] + ")");
                //System.out.println("Prev Sigmoid params: (" + activationList[k].params_prev[0] + ")");
            }
        }
    }

    public void resetGradientValues() {
        premiseParamCnt = 0;
        consequtiveParamCnt = 0;
        for (int k = 0; k < activationCnt; k++) {
            for (int n = 0; n < activationList[k].params.length; n++) {
                activationList[k].params_prev[n] = activationList[k].params[n];
                activationList[k].params_delta[n] = 0.0;
                premiseParamCnt++;
            }
        }
        for (int k = 0; k < linearParamCnt; k++) {
            linearParamsPrev[k] = linearParams[k];
            linearParamGrads[k] = 0.0;
            consequtiveParamCnt++;
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
    public double[] forwardPass(double[] inputs, int tillLayerID, boolean bVerbose) {
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
        ruleVals = new double[ruleList.length];
        double ruleSum = 0.0;
        for (int i = 0; i < ruleList.length; i++) {
            ruleSum += ruleList[i].calculate(activationList);
            ruleVals[i] = ruleList[i].getRuleVal();
            if (bVerbose)
                System.out.print("" + ruleVals[i] + " ");
        }

        if (bVerbose)
            System.out.println();

        if (tillLayerID == 2)
            return ruleVals;

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
            funcVals[i] = 0.0;
            /*
             //MF to Defuzz
            for (int j = 0; j < activationList.length; j++) {
                funcVals[i] += linearParams[i * (activationList.length + 1) + j] * activationList[j].getActivationVal();
            }
            // add bias parameter
            funcVals[i] += linearParams[i * (activationList.length + 1) + activationList.length];
             */

            for (int j = 0; j < inputs.length; j++) {
                funcVals[i] += linearParams[i * (inputs.length + 1) + j] * inputs[j];
            }
            // add bias parameter
            funcVals[i] += linearParams[i * (inputs.length + 1) + inputs.length];

            defuzzVals[i] = funcVals[i] * normalizedVals[i];
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
            //layerOutput[0] += normalizedVals[i];
        }
        if (bVerbose)
            System.out.println("" + layerOutput[0]);
        return layerOutput;
    }

    void adamLearning(int batchSize, int epochCnt, double minError, double[][] inputs, double[] outputs, boolean bVisualize, boolean bDebug) {
        double alpha = 0.0001;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 10e-8;
        double[] errors = new double[epochCnt];
        double maxError = 0.0;
        GraphPanel graphPanel = new GraphPanel();

        // save old activation values for visualization
        Activation[] oldActivations = activationList.clone();
        for (int i = 0; i < activationCnt; i++)
            oldActivations[i] = activationList[i].clone();

        if (bVisualize) {
            JFrame frame = new JFrame("ANFIS Adam Learning");
            frame.setSize(600, 300);
            frame.add(graphPanel);
            frame.setVisible(true);
        }

        // Calculate the initial Error
        double initError = 0.0;
        for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
            // pass till the end and calculate output value
            double[] outputValue = forwardPass(inputs[recIdx], -1, false);
            initError += Math.pow(outputs[recIdx] - outputValue[0], 2) / 2;
        }
        System.out.println("Error before training: " + initError);

        // iteration count
        int iterCnt = 1;
        // set to maximum to satisfy the (errors[0] > minError) check below (in "while")
        errors[0] = Double.MAX_VALUE;

        double[] m = new double[premiseParamCnt + consequtiveParamCnt];
        double[] v = new double[premiseParamCnt + consequtiveParamCnt];
        int t = 0;

        // repeat until a) error is minimized, b) max epoch count is reached or c) alpha range is too small
        while (iterCnt < epochCnt) {
            // Iterate through all training samples
            for (int recIdx = 0; recIdx < inputs.length; recIdx+=batchSize) {

                // 1. store activation parameters in a temporary params_prev.
                // 2. reset field for gradient
                resetGradientValues();

                // Mini-batch procedure
                for (int batchIdx = recIdx; batchIdx < Math.min(recIdx+batchSize,inputs.length); batchIdx++) {
                    calculateGradient(inputs[batchIdx], outputs[batchIdx], bDebug);
                }

                t++;
                int paramIdx = 0;
                for (int k = 0; k < activationCnt; k++)
                    for (int n = 0; n < activationList[k].params.length; n++) {
                        double g = activationList[k].params_delta[n]/batchSize;
                        m[paramIdx] = beta1 * m[paramIdx] + (1 - beta1) * g;
                        v[paramIdx] = beta2 * v[paramIdx] + (1 - beta2) * Math.pow(g, 2);
                        double m_hat = m[paramIdx] / (1 - Math.pow(beta1, t));
                        double v_hat = v[paramIdx] / (1 - Math.pow(beta2, t));
                        activationList[k].params[n] = activationList[k].params[n] - alpha * m_hat / (Math.sqrt(v_hat) + eps);
                        paramIdx++;
                    }
                for (int n = 0; n < linearParamCnt; n++) {
                    double g = linearParamGrads[n]/batchSize;
                    m[paramIdx] = beta1 * m[premiseParamCnt + n] + (1 - beta1) * g;
                    v[paramIdx] = beta2 * v[premiseParamCnt + n] + (1 - beta2) * Math.pow(g, 2);
                    double m_hat = m[paramIdx] / (1 - Math.pow(beta1, t));
                    double v_hat = v[paramIdx] / (1 - Math.pow(beta2, t));
                    linearParams[n] = linearParams[n] - alpha * m_hat / (Math.sqrt(v_hat) + eps);
                    paramIdx++;
                }
            }

            errors[iterCnt] = 0.0;

            for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
                // pass till the end and calculate output value
                double[] outputValue = forwardPass(inputs[recIdx], -1, false);
                errors[iterCnt] += Math.pow(outputs[recIdx] - outputValue[0], 2) / 2;
            }

            maxError = Math.max(maxError, errors[iterCnt]);

            graphPanel.setData(maxError, errors);
            System.out.println("\nEpoch = " + iterCnt + "; Error = " + errors[iterCnt]);
            if (errors[iterCnt] / inputs.length < minError)
                break;

            iterCnt++;
        }

        // Print parameters of Membership Functions after learning
        printActivationParams("Final");

        // Visualize Membership functions (show initial and updated MFs)
        if (bVisualize) {
            JFrame[] mfFrame = new JFrame[activationCnt];
            int frameWidth = 200;
            int framwHeight = 150;
            int horizWndCnt = 4; // count of windows in one horizontal line
            int pad = 20; // space between adjacent windows
            for (int k = 0; k < activationCnt; k++) {
                // Draw graph in [-10,10] range (to see how it looks like) but outline behaviour in our [-1,1] range
                MFGraph mfg = new MFGraph(activationList[k], oldActivations[k], -3, 3, -1, 1);
                mfFrame[k] = new JFrame("Activation " + (k + 1));
                mfFrame[k].setSize(frameWidth, framwHeight);
                mfFrame[k].setLocation((k % horizWndCnt) * frameWidth + pad, (k / horizWndCnt) * framwHeight + pad);
                mfFrame[k].add(mfg);
                mfFrame[k].setVisible(true);
            }
        }

    }

    /**
     * @param bHybrid    true if Hybrid Learning algorithm is needed, otherwise, just  backpropogation
     * @param epochCnt   count of maximum epochs (iterations over training set)
     * @param minError   minimum error to stop
     * @param inputs     all training inputs
     * @param outputs    desired outputs
     * @param bVisualize is visualization needed
     * @param bDebug     is additional output needed
     */
    void startTraining(boolean bHybrid, int epochCnt, double minError, double[][] inputs, double[] outputs, boolean bVisualize, boolean bDebug) {
        // used to reset parameters when value is NaN
        double[] errors = new double[epochCnt];
        double maxError = 0.0;
        double alpha = 0.0;
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

        if (bDebug) {
            // Runs Sequental LSE in batch mode to find consequent parameters
            System.out.println("Initial Linear Params:");
            for (int lp = 0; lp < linearParamCnt; lp++)
                System.out.println("Linear[" + lp + "]=" + linearParams[lp]);
        }

        // Calculate the initial Error
        double initError = 0.0;
        for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
            // pass till the end and calculate output value
            double[] outputValue = forwardPass(inputs[recIdx], -1, false);
            initError += Math.pow(outputs[recIdx] - outputValue[0], 2) / 2;
        }
        System.out.println("Error before training: " + initError);

        // iteration count
        int iterCnt = 0;
        // set to maximum to satisfy the (errors[0] > minError) check below (in "while")
        errors[0] = Double.MAX_VALUE;

        // repeat until a) error is minimized, b) max epoch count is reached or c) alpha range is too small
        while (iterCnt < epochCnt) {
            if (bHybrid) {
                System.out.println("Hybrid learning is on");
                // This matrix stores input information for the LSE learning.
                // It stores the input of the defuzzification layer - output of the normalization layer and inputs to the ANFIS
                double[][] A = new double[inputs.length][linearParamCnt];

                for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
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

                LSE_Optimization lse = new LSE_Optimization();
                double[] linearP = lse.findParameters(A, outputs);
                linearParams = linearP;
            }

            if (bDebug) {
                System.out.println("Linear Params:");
                for (int lp = 0; lp < linearParamCnt; lp++)
                    System.out.println("Linear[" + lp + "]=" + linearParams[lp]);
            }

            // --- Run BACK PROPOGATION ---
            if (bDebug)
                printActivationParams("New epoch");

            // 1. store activation parameters in a temporary params_prev.
            // 2. reset field for gradient
            resetGradientValues();

            if (bDebug) {
                // print delimeter
                System.out.println("____________________________________________");
            }
            // Iterate through all training samples
            for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
                calculateGradient(inputs[recIdx], outputs[recIdx], bDebug);
            }

            // and sum all gradients of each parameter
            double grad_norm = 0.0;
            for (int k = 0; k < activationCnt; k++)
                for (int n = 0; n < activationList[k].params.length; n++) {
                    grad_norm += Math.pow(activationList[k].params_delta[n], 2);
                    if (bDebug)
                        System.out.println("Grad[" + k + "][" + n + "] = " + activationList[k].params_delta[n]);
                }
            if (!bHybrid) {
                for (int n = 0; n < linearParamCnt; n++)
                    grad_norm += Math.pow(linearParamGrads[n], 2);
            }

            grad_norm = Math.sqrt(grad_norm);
            System.out.println("Gradient norm: " + grad_norm);

            // find the range for Golden Section Rule each Nth iteration.
            alpha = calculateAlpha(bHybrid, grad_norm, inputs, outputs, false);
            //alpha = 0.01;
            System.out.println("Alpha is : " + alpha);

            // Adjust weights according to the minimal error
            adjustMFweights(alpha, grad_norm);
            if (!bHybrid) {
                adjustConsequentWeights(alpha, grad_norm);
            }

            errors[iterCnt] = 0.0;
            for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
                // pass till the end and calculate output value
                double[] outputValue = forwardPass(inputs[recIdx], -1, false);
                errors[iterCnt] += Math.pow(outputs[recIdx] - outputValue[0], 2) / 2;
            }

            maxError = Math.max(maxError, errors[iterCnt]);

            graphPanel.setData(maxError, errors);
            System.out.println("\nEpoch = " + iterCnt + "; Error = " + errors[iterCnt]);
            if (errors[iterCnt] / inputs.length < minError)
                break;

            iterCnt++;
        }
        // Print parameters of Membership Functions after learning
        printActivationParams("Final");

        // Visualize Membership functions (show initial and updated MFs)
        if (bVisualize) {
            JFrame[] mfFrame = new JFrame[activationCnt];
            int frameWidth = 200;
            int framwHeight = 150;
            int horizWndCnt = 4; // count of windows in one horizontal line
            int pad = 20; // space between adjacent windows
            for (int k = 0; k < activationCnt; k++) {
                // Draw graph in [-10,10] range (to see how it looks like) but outline behaviour in our [-1,1] range
                MFGraph mfg = new MFGraph(activationList[k], oldActivations[k], -3, 3, -1, 1);
                mfFrame[k] = new JFrame("Activation " + (k + 1));
                mfFrame[k].setSize(frameWidth, framwHeight);
                mfFrame[k].setLocation((k % horizWndCnt) * frameWidth + pad, (k / horizWndCnt) * framwHeight + pad);
                mfFrame[k].add(mfg);
                mfFrame[k].setVisible(true);
            }
        }

    }

    private double calculateAlpha(boolean bHybrid, double grad_norm, double[][] inputs, double[] outputs, boolean bDebug) {
        double eps = 0.000001;
        double globalA = 0.0;
        double globalB = -1.0;
        double[] alpha = new double[2];
        int minErrorIdx = 0;


        // search for the range to be used in "Golden Section Rule".
        double[] err = new double[2];
        err[0] = Double.MAX_VALUE;

        double step = 0;
        int n = 1;

        // just make 20 steps
        while (globalB < 0) {
            adjustMFweights(step, grad_norm);
            if (!bHybrid) {
                adjustConsequentWeights(step, grad_norm);
            }

            // calculate output error
            err[1] = 0.0;
            for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
                double[] outputValue = forwardPass(inputs[recIdx], -1, false);
                err[1] += Math.pow(outputs[recIdx] - outputValue[0], 2) / 2;
            }
            //printActivationParams("alpha = " + step);
            if (bDebug)
                System.out.println(step + ":" + err[1]);

            if ((globalB < 0.0) && (err[1] < err[0])) {
                globalA = step;
                err[0] = err[1];
                err[1] = 0.0;
            } else if (err[1] >= err[0]) {
                globalB = step;
            }

            step += 1;
        }

        if (bDebug)
            System.out.println("Golden Section range: [" + globalA + "," + globalB + "]");

        double a = globalA - 1;
        double b = globalB;
        err = new double[2];

        while (Math.abs(b - a) > eps) {
            alpha[0] = b - (b - a) / 1.618;
            alpha[1] = a + (b - a) / 1.618;

            for (int aidx = 0; aidx < alpha.length; aidx++) {
                // adjust parameters
                adjustMFweights(alpha[aidx], grad_norm);
                if (!bHybrid) {
                    adjustConsequentWeights(alpha[aidx], grad_norm);
                }

                // calculate output error
                err[aidx] = 0.0;
                for (int recIdx = 0; recIdx < inputs.length; recIdx++) {
                    double[] outputValue = forwardPass(inputs[recIdx], -1, false);
                    err[aidx] += Math.pow(outputs[recIdx] - outputValue[0], 2) / 2;
                }
                if (bDebug)
                    System.out.print("E[" + String.format("%.06f", alpha[aidx]) + "]=" + String.format("%.06f", err[aidx]) + "; ");
            }
            if (bDebug)
                System.out.println();

            if (err[0] > err[1]) {
                a = alpha[0];
                minErrorIdx = 1;
            } else {
                b = alpha[1];
                minErrorIdx = 0;
            }
        }
        return alpha[minErrorIdx];
    }


    /**
     * calculate gradient for the given sample
     *
     * @param
     */
    public Activation[] calculateGradient(double[] inputs, double output, boolean bVerbose) {
        // pass till the end and calculate output value
        double[] outputValue = forwardPass(inputs, -1, false);
        double diff = (output - outputValue[0]);

        if (bVerbose) {
            System.out.println(String.format("%30s", "\n\ncalculateGradient()"));
            System.out.println(String.format(" %76s ", "(Desired - Output) = " + diff));
        }

        // Calculate the derivatives of the CONSEQUENT parameters
        for (int k = 0; k < defuzzVals.length; k++) {
            for (int z = 0; z < inputCnt; z++)
                linearParamGrads[k * (inputCnt + 1) + z] += -1 * diff * (normalizedVals[k] * inputs[z]);
            linearParamGrads[k * (inputCnt + 1) + inputCnt] += -1 * diff * normalizedVals[k];
/*
            // MF to Defuzz
            for (int z = 0; z < activationCnt; z++)
                linearParamGrads[k * (activationCnt+1) + z] += -1 * diff * (normalizedVals[k] * activationList[z].activationVal);
            linearParamGrads[k * (activationCnt+1) + activationCnt] += -1 * diff * normalizedVals[k];
*/
        }

        // calculate Error->Output->Defuzz->Normalization gradients
        if (bVerbose) {
            System.out.println("\nCalculated GRADIENTS");
            System.out.println("Final layer(5)   Defuz. Layer(4)  Norm. Layer(3) ");
        }

        // Calculate the derivatives of the PREMISE parameters
        for (int k = 0; k < defuzzVals.length; k++) {
            normalizedGrads[k] = -1 * diff * funcVals[k];
            if (bVerbose) {
                System.out.print("F(" + k + ")=" + String.format("%.6f", funcVals[k]) + "; ");
                System.out.print("  D[" + k + "]=" + String.format("%.6f", (-1 * diff)) + "; ");
                System.out.println("  N[" + k + "]=" + String.format("%.6f", normalizedGrads[k]) + "; ");
            }
        }

        if (bVerbose) {
            System.out.println();
        }

        // calculate Normalization->Rules gradients
        if (bVerbose) {
            System.out.println("---------Rule layer(2)----------");
        }
        double ruleSum = 0.0;
        for (int k = 0; k < ruleList.length; k++) {
            ruleSum += ruleVals[k];
        }

        for (int k = 0; k < ruleList.length; k++) {
            // Iterate over each "Rule<->Normalization" connection
            ruleList[k].gradientVal = 0.0;
            if (bVerbose) {
                System.out.println(String.format(" %70s ", "For rule #" + k + " with each layer 4 node separately:"));
            }
            for (int m = 0; m < normalizedVals.length; m++) {
                double ruleGrad = 0.0;

                if (k == m)
                    ruleGrad = normalizedVals[m] * (1 - normalizedVals[m]) / ruleVals[k];
                else
                    ruleGrad = (-1) * Math.pow(normalizedVals[m], 2) / ruleVals[m];
                //ruleGrad =  (-1) * normalizedVals[m] / ruleSum; // above is simplified

                if (bVerbose)
                    System.out.print("r[" + k + "," + m + "]=" + String.format("%.6f", ruleGrad) + "; ");

                ruleList[k].gradientVal += normalizedGrads[m] * ruleGrad;
            }
            if (bVerbose) {
                System.out.println(String.format("\n %50s ", "Total (summarized) for rule #" + k));
                System.out.println("R[" + k + "]=" + String.format("%.6f", ruleList[k].gradientVal) + "; ");
            }
        }


        // calculate gradients for each membership function (Rules->Membership)
        // first reset all values
        for (int k = 0; k < activationCnt; k++) {
            activationList[k].gradientVal = 0.0;
        }

        if (bVerbose) {
            System.out.println("         Activation layer(1)");
        }
        for (int k = 0; k < ruleList.length; k++) {
            // find all connected membership functions to each rule and summarize the value
            for (int m = 0; m < ruleList[k].inputActivations.length; m++) {
                int idx = ruleList[k].inputActivations[m];

                activationList[idx].gradientVal += ruleList[k].gradientVal * ruleList[k].getRuleVal() / activationList[idx].activationVal;
                if (bVerbose)
                    System.out.print("A[" + idx + "]=" + String.format("%.6f", activationList[idx].gradientVal) + "; ");
            }
        }

        if (bVerbose) {
            System.out.println();
            System.out.println();
        }

        // Now, find final gradients!
        for (int k = 0; k < activationCnt; k++) {
            if (activationList[k].mf == Activation.MembershipFunc.BELL) {
                // derivatives A,B and C
                double ad, bd, cd;

                ad = activationList[k].calcDerivative(inputs, 1);
                bd = activationList[k].calcDerivative(inputs, 2);
                cd = activationList[k].calcDerivative(inputs, 3);

                activationList[k].params_delta[0] += ad * activationList[k].gradientVal;
                activationList[k].params_delta[1] += bd * activationList[k].gradientVal;
                activationList[k].params_delta[2] += cd * activationList[k].gradientVal;
            }
            else if (activationList[k].mf == Activation.MembershipFunc.CENTERED_BELL) {
                    // derivatives A,B and C
                    double bd, cd;

                    bd = activationList[k].calcDerivative(inputs, 1);
                    cd = activationList[k].calcDerivative(inputs, 2);

                    activationList[k].params_delta[0] += bd * activationList[k].gradientVal;
                    activationList[k].params_delta[1] += cd * activationList[k].gradientVal;
            } else if (activationList[k].mf == Activation.MembershipFunc.SIGMOID) {
                // derivatives A
                double ad;

                ad = activationList[k].calcDerivative(inputs, 1);
                activationList[k].params_delta[0] += ad * activationList[k].gradientVal;
            }
        }
        return activationList;
    }

    /**
     * Adjusts the parameters of the Membership Functions according to the alpha parameter.
     *
     * @param alpha    coefficient that calculated as 1D-minimisation
     * @param gradNorm the norm (length) of the gradient
     */

    void adjustMFweights(double alpha, double gradNorm) {
        for (int k = 0; k < activationCnt; k++)
            for (int n = 0; n < activationList[k].params.length; n++)
                activationList[k].params[n] = activationList[k].params_prev[n] - alpha * activationList[k].params_delta[n] / (gradNorm);
    }

    void adjustConsequentWeights(double alpha, double gradNorm) {
        for (int i = 0; i < linearParamCnt; i++)
            linearParams[i] = linearParamsPrev[i] - alpha * linearParamGrads[i] / (gradNorm);
    }

}

class Input {
    Activation activation;
}