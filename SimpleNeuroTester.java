package mathcomp.oletsky.neuro;

import mathcomp.oletsky.mathhelper.VectMatr;

import java.util.Arrays;

/**
 * @author O.Oletsky
 * Training a simple neuron
 * to recognize conjunctions, disjunctions etc.
 * But it can't be trained for XOR
 */
public class SimpleNeuroTester {
    public static void main(String[] args) {
        SimpleNeuron neuron = new SimpleNeuron(
                new TresholdFunction()
        );

        String fName="orsimpleperceptron.txt";

        //Training set
        double[][] trainingInputs={
                {0., 0., 1.},
                {0., 1., 1.},
                {1., 0., 1.},
                {1., 1., 1.}
        };
        double[] trainingDisjOutputs = {0., 1., 1., 1.};
        double[] trainingConjOutputs = {0., 0., 0., 1.};
        double[] trainingXOutputs = {0., 0., 1., 1.};
        double[] trainingXOROutputs = {0., 1., 1., 0.};

        double[] trainingOutputs = trainingDisjOutputs;

        //Training
        final double EPS = 1.E-5;
        final double GAMMA = 0.5;
        final int MAX_ITERS=2000000;

        boolean success=neuron.train(trainingInputs,
                trainingOutputs,
                EPS,
                GAMMA,
                MAX_ITERS);

        if (success)
            System.out.println("Neuron has been successfully trained");
        else
            System.out.println("Training was not very successful");
        neuron.save(fName);
        System.out.println("Configuration saved to file "+fName);

        //Testing
        double[][] controlSet = {
                {0., 0., 1.},
                {0., 1., 1.},
                {1., 0., 1.},
                {1., 1., 1.}
        };

        double[] testResults=new double[controlSet.length];
        for (int i=0; i<controlSet.length; i++) {
            double[] test = controlSet[i];
            testResults[i] = neuron.act(test);
        }

        System.out.println("Test results");
        VectMatr.defaultOutputVector(testResults);
    }
}
