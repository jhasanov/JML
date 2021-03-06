package jml.anfis;

import javax.swing.*;
import java.awt.*;

/**
 * Created by Jamal Hasanov on 6/22/2017.
 */
public class MFGraph extends JPanel {
    Activation mfFunc, mfBefore;
    int horizPad = 10;
    int vertPad = 10;
    double xMin = 0;
    double xMax = 0;
    double valueMin = 0;
    double valueMax = 0;

    /**
     *
     * @param mf Membership Function
     * @param xMin minimum X value of the drawing range
     * @param xMax maximum X value of the drawing range
     * @param vmin minimum value that your data gets (inputs to this activationMF). Should be >= xMin
     * @param vmax maximum value that your data gets (inputs to this activationMF). Should be <= xMax
     * @param mfBefore Activation parameters before the training (initial parameters)
     */
    public MFGraph(Activation mf, Activation mfBefore, double xMin, double xMax,double vmin, double vmax) {
        mfFunc = mf;
        this.xMin = xMin;
        this.xMax = xMax;
        this.valueMin =vmin;
        this.valueMax = vmax;
        this.mfBefore = mfBefore;
    }

    @Override
    protected void paintComponent(Graphics gr) {
        super.setBackground(Color.white);
        super.paintComponent(gr);
        Graphics2D gr2 = (Graphics2D)gr;
        gr2.setColor(Color.BLUE);
        Dimension dim = getSize();

        gr2.setColor(Color.BLACK);

        /*
        if (mfFunc.mf == Activation.MembershipFunc.SIGMOID)
            gr2.drawString("SIGMOID",horizPad,20);
        else if (mfFunc.mf == Activation.MembershipFunc.BELL)
            gr2.drawString("BELL",horizPad,20);
        else if (mfFunc.mf == Activation.MembershipFunc.CENTERED_BELL)
            gr2.drawString("CENTERED BELL",horizPad,20);
        gr2.setColor(Color.DARK_GRAY);
        gr2.drawString("Initial activationMF",horizPad,40);
        gr2.setColor(Color.BLUE);
        gr2.drawString("Updated activationMF",horizPad,60);
        gr2.setColor(Color.MAGENTA);
        gr2.drawString("Intput data range",horizPad,80);
        */
        //gr2.setColor(Color.BLUE);

        //gr2.drawString(""+xMin,horizPad,dim.height-vertPad);
        //gr2.drawString(""+xMax,dim.width-horizPad,dim.height-vertPad);
        // Draw X and Y axis
        gr2.drawLine(0, dim.height-1, dim.width, dim.height-1);
        gr2.drawLine(dim.width/2, 0, dim.width/2,dim.height);

        double horizSteps, vertSteps;
        horizSteps = (dim.width - horizPad * 2) / (xMax - xMin);
        vertSteps = (dim.height - vertPad * 2);
        int prevX=-1;
        int prevY=-1;
        int prevCurrY = -1;

        if (mfFunc != null) {
            for (int i = 0; i < dim.width - 2 * horizPad; i++) {

                /*
                // draw input data range:
                int leftInputRange = (int) (horizPad+(valueMin-xMin)*horizSteps);
                int rightInputRange = (int) (horizPad+(valueMax-xMin)*horizSteps);
                gr2.setColor(Color.MAGENTA);
                float [] dash = {5.0f};
                Stroke dashedStroke = new BasicStroke(1.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER,
                        10.0f, dash, 0.0f);
                gr2.setStroke(dashedStroke);
                gr2.drawLine(leftInputRange,vertPad,leftInputRange,(int)(dim.getHeight()-vertPad));
                gr2.drawLine(rightInputRange,vertPad,rightInputRange,(int)(dim.getHeight()-vertPad));
                */

                gr2.setStroke(new BasicStroke(1.0f));
                int sigValCurr = 0;
                int sigValInit = 0;
                if (mfFunc.mf == Activation.MembershipFunc.SIGMOID) {
                    sigValInit = (int) (mfBefore.sigmoidFunc(xMin + i / horizSteps) * vertSteps);
                    sigValCurr = (int) (mfFunc.sigmoidFunc(xMin + i / horizSteps) * vertSteps);
                    }
                else if (mfFunc.mf == Activation.MembershipFunc.BELL) {
                    sigValInit = (int) (mfBefore.bellFunc(xMin + i / horizSteps) * vertSteps);
                    sigValCurr = (int) (mfFunc.bellFunc(xMin + i / horizSteps) * vertSteps);
                    }
                else if (mfFunc.mf == Activation.MembershipFunc.CENTERED_BELL) {
                    sigValInit = (int) (mfBefore.centeredBellFunc(xMin + i / horizSteps) * vertSteps);
                    sigValCurr = (int) (mfFunc.centeredBellFunc(xMin + i / horizSteps) * vertSteps);
                }
                int x = horizPad + i;
                int origY = dim.height - (vertPad + sigValInit);
                int currY = dim.height - (vertPad + sigValCurr);

                // draw activation function with initial parameters
                if (prevX != -1) {
                    gr2.setColor(Color.LIGHT_GRAY);
                    //gr2.drawRect(x, y, 1, 1);
                    gr2.setStroke(new BasicStroke(2.0f));
                    gr2.drawLine(prevX,prevY,x,origY);

                    // draw current activation function
                    //if (( i > (valueMin-xMin)*horizSteps) && ( i < (valueMax-xMin)*horizSteps) )
                    //    gr2.setColor(Color.MAGENTA);
                    //else
                    gr2.setStroke(new BasicStroke(1.0f));
                    gr2.setColor(Color.BLACK);
                    //gr2.setStroke(new BasicStroke(1.0f));
                    gr2.drawLine(prevX,prevCurrY,x,currY);
                    //gr2.drawRect(x, y, 1, 1);
                }
                prevX = x; prevY = origY; prevCurrY = currY;
            }
        }

    }
}
