package jml.anfis;

public class Rule {
    public enum RuleOperation {AND, OR};
    RuleOperation oper;
    int[] inputActivations;
    double ruleVal;
    // the value of the calculated gradient till this node
    double gradientVal = 0.0;

    public Rule(int[] inputMFidx, RuleOperation oper) {
        this.inputActivations = inputMFidx;
        this.oper = oper;
    }

    public RuleOperation getOper() {
        return oper;
    }

    public void setOper(RuleOperation oper) {
        this.oper = oper;
    }

    public int[] getInputActivations() {
        return inputActivations;
    }

    public void setInputActivations(int[] inputActivations) {
        this.inputActivations = inputActivations;
    }

    public void setRuleVal(double ruleVal) {
        this.ruleVal = ruleVal;
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
