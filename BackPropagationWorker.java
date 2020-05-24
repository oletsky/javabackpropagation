package mathcomp.oletsky.neuro;

import mathcomp.oletsky.mathhelper.VectMatr;

public class BackPropagationWorker {
    public static void main(String[] args) {
        final String fName="ornetwork.txt";
        BackPropagationNetwork network = BackPropagationNetwork.load(fName);
        System.out.println("Network loaded");
        //Testing
        double[][] controlSet = {
                {0., 0.},
                {0., 1.},
                {1., 0.},
                {1., 1.}
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
