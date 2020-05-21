package mathcomp.oletsky.neuro;

import mathcomp.oletsky.mathhelper.VectMatr;

import java.util.Random;

public class Neuron {
    ActivationFunction activationFunction;
    private double[] weights;
    private int kolInputs;

    private double output;

    public Neuron(int kolInputs,
                  ActivationFunction activationFunction) {
        this.kolInputs = kolInputs;
        this.activationFunction = activationFunction;
        weights = new double[kolInputs+1];
        //Random setting
        Random r = new Random();
        for (int i =0; i<=kolInputs; i++) {
            weights[i]=r.nextDouble();
        }

    }

    public int getInputCount() {
        return this.kolInputs;
    }

    public double getOutput() {
        return this.output;
    }

    public double getWeight(int j) {
        return this.weights[j];
    }

    public double[] getWeights() {
        return this.weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setWeight(int k, double value) {
        this.weights[k]=value;
    }

    public double act(double[] inputs) {
        this.proceed(inputs);
        return this.output;
    }

    public void proceed(double[] inputs) {
        if (inputs.length+1!=weights.length) {
            throw new IllegalArgumentException("Dimensions of inputs and weights are differemt!");
        }
        double prod = VectMatr.calculateScalarProduct
                (Neuron.expandWithOne(inputs),
                weights);
        this.output = activationFunction.activate(prod);
    }

    public static double[] expandWithOne(double[] input) {
        double[] res = new double[input.length+1];
        for (int i=0; i<input.length; i++) {
            res[i]=input[i];
        }
        res[input.length]=1.;
        return res;

    }
}
