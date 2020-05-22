package mathcomp.oletsky.neuro;

import mathcomp.oletsky.mathhelper.VectMatr;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class SimpleNeuron {
    private double[] weights;
    private ActivationFunction activFunction;

    public SimpleNeuron(ActivationFunction activFunction) {
        this.activFunction = activFunction;
    }

    public SimpleNeuron(double[] weights,
                        ActivationFunction activFunction) {
        this.weights = weights;
        this.activFunction = activFunction;
    }

    public double act(double[] inputs) {
        if (inputs.length != weights.length) {
            throw new RuntimeException("Wrong dimensions");
        }
        double r = VectMatr.calculateScalarProduct(weights, inputs);
        return activFunction.activate(r);
        }

    public boolean train(double[][] inputSet,
                      double[] outputSet,
                      double EPS,
                      double gamma,
                         int MAX_ITERS
    ) {
        boolean forcedExit=false;

        if (inputSet.length != outputSet.length) {
            throw new RuntimeException("Wrong training dimensions");
        }
        weights = new double[inputSet[0].length];

        int numbEpochs = 0;


        boolean changed = true;
        while (changed) {
            numbEpochs++;
            if (numbEpochs==MAX_ITERS) {
                forcedExit=true;
                break;
            }
            changed=false;
            for (int k = 0; k < inputSet.length; k++) {
                double output = act(inputSet[k]);
                double err = outputSet[k] - output;
                if (Math.abs(err)>EPS) {
                    changed=true;
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += err * gamma * inputSet[k][j];

                    }
                }
            }

        }
        System.out.println("There were "+numbEpochs+" epochs");
        return !forcedExit;
    }

    public void save (String fName){
        try (PrintWriter pw = new PrintWriter(fName)) {
            pw.println(activFunction.inform());
            for (int i=0; i<weights.length; i++) {
                pw.printf("%10.5f",weights[i]);
                if (i!=weights.length-1) pw.print(";");

            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
