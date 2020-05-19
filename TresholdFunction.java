package mathcomp.oletsky.neuro;

public class TresholdFunction implements ActivationFunction {

    @Override
    public double activate(double s) {
        if (s >= 0.5) return 1.;
        else return 0.;
    }
}
