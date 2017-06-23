package jml.anfis;

import javax.swing.*;
import java.awt.*;

/**
 * Created by itjamal on 6/22/2017.
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
     * @param vmin minimum value that your data gets (inputs to this MF). Should be >= xMin
     * @param vmax maximum value that your data gets (inputs to this MF). Should be <= xMax
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
        super.paintComponent(gr);
        gr.setColor(Color.BLUE);
        Dimension dim = getSize();

        gr.setColor(Color.BLUE);

        if (mfFunc.mf == Activation.MembershipFunc.SIGMOID)
            gr.drawString("SIGMOID",horizPad,20);
        else if (mfFunc.mf == Activation.MembershipFunc.BELL)
            gr.drawString("BELL",horizPad,20);
        gr.setColor(Color.DARK_GRAY);
        gr.drawString("Initial MF",horizPad,40);
        gr.setColor(Color.BLUE);
        gr.drawString("All range",horizPad,60);
        gr.setColor(Color.MAGENTA);
        gr.drawString("Intput data",horizPad,80);

        gr.setColor(Color.BLUE);
        gr.drawString(""+xMin,horizPad,dim.height-vertPad);
        gr.drawString(""+xMax,dim.width-horizPad,dim.height-vertPad);

        double horizSteps, vertSteps;
        horizSteps = (dim.width - horizPad * 2) / (xMax - xMin);
        vertSteps = (dim.height - vertPad * 2);


        if (mfFunc != null) {
            for (int i = 0; i < dim.width - 2 * horizPad; i++) {
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
                int x = horizPad + i;
                int y = dim.height - (vertPad + sigValInit);

                // draw activation function with initial parameters
                gr.setColor(Color.DARK_GRAY);
                gr.drawRect(x, y, 1, 1);

                // draw current activation function
                y = dim.height - (vertPad + sigValCurr);
                if (( i > (valueMin-xMin)*horizSteps) && ( i < (valueMax-xMin)*horizSteps) )
                    gr.setColor(Color.MAGENTA);
                else
                    gr.setColor(Color.BLUE);
                gr.drawRect(x, y, 1, 1);
            }
        }

    }
}
