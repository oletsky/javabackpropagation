package mathcomp.oletsky.neuro;

import mathcomp.oletsky.mathhelper.VectMatr;

public class BackPropagationTester {
    public static void main(String[] args) {
        //Setting parameters

        final int kolInputs = 2;
        ActivationFunction[] activeFunctions = {
                new SigmoidFunction(),
                new SigmoidFunction(),
                new SigmoidFunction()

        };
        int[] kolNeurons = {5, 10, 1};

        //Creating network
        BackPropagationNetwork network =
                new BackPropagationNetwork(
                        kolInputs,
                        activeFunctions,
                        kolNeurons);


        //Training set
        double[][] trainingInputs = {
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.}
        };
        double[][] trainingDisjOutputs =
                {
                        {0.},
                        {1.},
                        {1.},
                        {1.}
                };
        double[][] trainingConjOutputs = {
                {0.},
                {0.},
                {0.},
                {1.}
        };
        double[][] trainingXOutputs = {
                {0.}, {0.}, {1.}, {1.}
        };
        double[][] trainingXOROutputs = {
                {0.}, {1.}, {1.}, {0.}
        };

        double[][] trainingOutputs = trainingXOROutputs;

        //Training
        final double EPS = 1.E-12;
        final double GAMMA = 0.5;
        final int MAX_ITERS = 2000000;

        boolean success = network.train(trainingInputs,
                trainingOutputs,
                EPS,
                GAMMA,
                MAX_ITERS);

        if (success)
            System.out.println("Neuron has been successfully trained");
        else
            System.out.println("Training was not very successful");
        String fName="network.txt";
        network.save(fName);
        System.out.println("Network has been saved to file "+fName);
        //Testing
        double[][] controlSet = {
                {0.2, 0.1},
                {0.1, 0.9},
                {0.8, 0.1},
                {0.9, 0.9}
        };

        double[][] testResults = new double[controlSet.length][];
        for (int i = 0; i < controlSet.length; i++) {
            double[] test = controlSet[i];
            testResults[i] = network.act(test);
        }

        System.out.println("Test results");
        VectMatr.defaultOutputMatrix(testResults);

    }
}
