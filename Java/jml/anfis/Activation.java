package jml.anfis;


/**
 * Created by Jamal Hasanov on 6/8/2017.
 */
public class Activation {
    enum MembershipFunc {SIGMOID, BELL}
    ;

    MembershipFunc mf;
    double[] params;
    double[] params_prev; // previous value of the parameter
    double[] temp_params;
    double[] params_delta; // sum of updates to the param
    int inputNo; // index of the input

    double activationVal = 0.0;
    // the value of the calculated gradient till this node
    double gradientVal = 0.0;

    public Activation() {

    }

    public Activation(int inputNo, MembershipFunc mf) {
        this.inputNo = inputNo;
        this.mf = mf;

        if (mf == MembershipFunc.SIGMOID) {
            params = new double[1];
            params_prev = new double[1];
            params_delta = new double[1];
            params[0] = Math.random();
            params_prev[0] = params[0];
            gradientVal = 0.0;
            activationVal = 0.0;
        } else if (mf == MembershipFunc.BELL) {
            params = new double[3];
            params_prev = new double[3];
            params_delta = new double[3];
            params[0] = Math.random();
            params[1] = Math.random();
            params[2] = Math.random();
            params_prev[0] = params[0];
            params_prev[1] = params[1];
            params_prev[2] = params[2];
            gradientVal = 0.0;
            activationVal = 0.0;
        }
    }

    @Override
    public Activation clone() {
        Activation newObj = new Activation(inputNo,mf);
        newObj.mf = mf;
        newObj.params = params.clone();

        return newObj;
    }

    public void setRandomParams() {
        if (mf == MembershipFunc.SIGMOID) {
            params = new double[1];
            params_prev = new double[1];
            params[0] = Math.random();
            params_prev[0] = params[0];
            gradientVal = 0.0;
            activationVal = 0.0;
        } else if (mf == MembershipFunc.BELL) {
            params = new double[3];
            params_prev = new double[3];
            params[0] = Math.random();
            params[1] = Math.random();
            params[2] = Math.random();
            params_prev[0] = params[0];
            params_prev[1] = params[1];
            params_prev[2] = params[2];
            gradientVal = 0.0;
            activationVal = 0.0;
        }
    }

    public double[] getParamDelta() {
        return params_delta;
    }

    public double getGradientVal() {
        return gradientVal;
    }

    public double sigmoidFunc(double x,double a) {
        return (1 / (1 + Math.exp(a*x)));
    }

    public double sigmoidFunc(double x) {
        return (1 / (1 + Math.exp(params[0]*x)));
    }

    /**
     * derivative of the sigmoid function
     *
     * @return value of the derivative
     */

    public double sigmoidFuncDerivA(double x, double a) {
        // add 0.001 to avoid zero derivative
        double retval = -1*Math.pow(sigmoidFunc(x,a),2)*Math.exp(a*x)*x + 0.00000001;
        if (Double.isNaN(retval)) {
            System.out.println("sigmoidFuncDerivA("+x+","+a+") returned NaN");
        }
        return retval;
    }

    public double bellFunc(double x, double a, double b, double c) {
        return (1/(1+Math.pow(Math.abs((x-a)/c),2*b)));
    }

    public double bellFunc(double x) {
        return (1/(1+Math.pow(Math.abs((x-params[0])/params[2]),2*params[1])));
    }

    public double bellFuncDerivA(double x, double a, double b, double c) {
        double bellVal = bellFunc(x,a,b,c);
        double modulusSign = 1.0;
        if (x != a)
         modulusSign = (Math.abs((x-a)/c)/((x-a)/c));

        // add 0.001 to avoid zero derivative
        double retval = Math.pow(bellVal,2) * (2*b /c) * Math.pow(Math.abs((x-a)/c),2*b-1)*modulusSign + 0.00000001;
        if (Double.isNaN(retval)) {
            System.out.println("bellFuncDerivA("+x+","+a+","+b+","+c+") returned NaN");
        }

        return retval;
    }

    public double bellFuncDerivB(double x, double a, double b, double c) {
        double bellVal = bellFunc(x,a,b,c);

        // Note: when (X==A), Math.log(Math.abs((x-a) /c)) term will return "Infinity"...
        if (x == a)
            x += 0.001;

        // add 0.001 to avoid zero derivative
        double retval =  -1*Math.pow(bellVal,2) * Math.log(Math.abs((x-a) /c)) * Math.pow(Math.abs((x-a)/c),2*b)*2 + 0.00000001;

        if (Double.isNaN(retval)) {
            System.out.println("bellFuncDerivB("+x+","+a+","+b+","+c+") returned NaN");
        }
        return retval;
    }

    public double bellFuncDerivC(double x, double a, double b, double c) {
        double bellVal = bellFunc(x,a,b,c);
        double modulusSign = 1.0;

        if (x!= a)
            modulusSign = (Math.abs((x-a)/c)/((x-a)/c));

        // add 0.001 to avoid zero derivative
        double retval = Math.pow(bellVal,2) * (2*b /(c*c)) * Math.pow(Math.abs((x-a)/c),2*b-1)*(x-a)*modulusSign + 0.00000001;

        if (Double.isNaN(retval)) {
            System.out.println("bellFuncDerivC("+x+","+a+","+b+","+c+") returned NaN");
        }
        return retval;
    }

    // Left-to-right activation function
    public double ascend(double x, double a, double b) {
        if ((x>=a) && (x<b)) {
            return (x-a)/(b-a);
        }
        else if (x>=b) {
            return 1;
        }
        else return 0;
    }

    // Trapezoidal-shaped membership function
    public double trap(double x,double a, double b, double c, double d) {
        if ((x>=0) && (x<a)) {
            return (x-a)/(b-a);
        }
        else if ((x>=a) && (x<b)) {
            return 1.0;
        }
        else if ((x>=b) && (x<c)) {
            return (d-x)/(d-c);
        }
        else
            return 0;
    }

    // execute activation function
    public double activate(double[] inputs) {
        if (mf == MembershipFunc.SIGMOID)
            activationVal = sigmoidFunc(inputs[inputNo],params[0]);
        else if (mf == MembershipFunc.BELL) {
            activationVal = bellFunc(inputs[inputNo], params[0], params[1], params[2]);
        }
        else
            activationVal = -1.0;

        return activationVal;
    }

    /** calculate derivative
        @param paramIdx index of the parameter that needs to be differentiated
     */
    public double calcDerivative(double[] inputs,int paramIdx) {
        double deriv = 0.0;
        if (mf == MembershipFunc.SIGMOID)
            deriv = sigmoidFuncDerivA(inputs[inputNo],params[0]);
        else if (mf == MembershipFunc.BELL) {
            if (paramIdx == 1)
                deriv = bellFuncDerivA(inputs[inputNo], params[0], params[1], params[2]);
            else if (paramIdx == 2)
                deriv = bellFuncDerivB(inputs[inputNo], params[0], params[1], params[2]);
            if (paramIdx == 3)
                deriv = bellFuncDerivC(inputs[inputNo], params[0], params[1], params[2]);
        }
        else
            deriv = 0;

        return deriv;
    }

    public double getActivationVal() {
        return activationVal;
    }
}
