package jml.anfis;

import javax.swing.*;
import java.awt.*;

/**
 * Created by itjamal on 6/17/2017.
 */
public class GraphPanel extends JPanel{
    double maxDataVal = 0.0;
    double [] dataValues;
    int horizPad = 10;
    int vertPad = 10;

    public GraphPanel() {
        super();
    }

    public GraphPanel(int horizPad, int vertPad) {
        this.horizPad = horizPad;
        this.vertPad = vertPad;
    }


    public void setData(double maxVal, double [] values) {
        maxDataVal = maxVal;
        dataValues = values;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics gr) {
        super.paintComponent(gr);
        gr.setColor(Color.BLUE);

        Dimension dim = getSize();
        double horizSteps, vertSteps;

        if (dataValues != null) {
            horizSteps = (dim.width - horizPad * 2) / dataValues.length;
            vertSteps = (dim.height - vertPad * 2) / maxDataVal;

            for (int i = 0; i < dataValues.length; i++) {
                int x = (int)(horizPad + i * horizSteps);
                int y = (int)(dim.height - (vertPad + dataValues[i] * vertSteps));
                gr.fillRect(x - 2, y-2, 4, 4);
                // show maximum first 5 digits
                String text = "";
                if (dataValues[i] != 0) {
                    text = "" + dataValues[i];
                    text = text.substring(0,Math.min(text.length(),5));
                }
                gr.drawString(text,x+5,y);
            }
        }
    }

}
