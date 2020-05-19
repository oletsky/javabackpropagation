package mathcomp.oletsky.neuro;

public class SigmoidFunction implements ActivationFunction{
    @Override
    public double activate(double s) {
        return 1./(1+Math.exp(-s));
    }
}
