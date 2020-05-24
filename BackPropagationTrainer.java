package mathcomp.oletsky.neuro;

import mathcomp.oletsky.mathhelper.VectMatr;

public class BackPropagationTrainer {
    public static void main(String[] args) {
        //Setting parameters

        final int kolInputs = 2;
        String[] funcNames = {"sigmoid"};

        int[] kolNeurons = {1};

        //Creating network
        BackPropagationNetwork network =
                new BackPropagationNetwork(
                        kolInputs,
                        funcNames,
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

        double[][] trainingOutputs = trainingDisjOutputs;

        //Training
        final double EPS = 1.E-10;
        final double GAMMA = 0.5;
        final int MAX_ITERS = 2000000;

        boolean success = network.train(trainingInputs,
                trainingOutputs,
                EPS,
                GAMMA,
                MAX_ITERS);

        if (success)
            System.out.println("Network has been successfully trained");
        else
            System.out.println("Training was not very successful");
        String fName="ornetwork.txt";
        network.save(fName);
        System.out.println("Network has been saved to file "+fName);

    }
}
