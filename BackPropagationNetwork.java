package mathcomp.oletsky.neuro;

import java.io.*;
import java.util.Arrays;

public class BackPropagationNetwork {
    private int kolLayers;
    private int kolInputs;
    private double[][] layerOutputs;
    private double[][] layerErrors;
    private Neuron[][] neurons;
    ActivationFunction[] activFunctions;

    public BackPropagationNetwork
            (int kolInputs,
             String[] funcNames,
             int[] kolNeuronsOnLayer){
        //Preparing infrastructure
        this.kolInputs=kolInputs;
        this.kolLayers=kolNeuronsOnLayer.length;
        this.activFunctions = new ActivationFunction[funcNames.length];
        for (int i=0; i<activFunctions.length; i++) {
            activFunctions[i]= loadFunctionByName(funcNames[i]);
        }
        //Creating network
        neurons = new Neuron[kolLayers][];
        for (int i=0; i<kolLayers; i++) {
            int kolNeuroInputs = (i==0)?
                    kolInputs:kolNeuronsOnLayer[i-1];
            neurons[i] = new Neuron[kolNeuronsOnLayer[i]];
            for (int j=0; j<kolNeuronsOnLayer[i]; j++) {
                neurons[i][j]=new Neuron(kolNeuroInputs,
                activFunctions[i]);

            }
        }

    }

    public double[] act(double[] inputs) {
        this.proceed(inputs);
        return this.getOutputs();
    }

    public void proceed(double[] inputs) {
        this.formLayerOutputs();
        this.formLayerErrors();

        //Actual proceeding
        for (int i=0; i<kolLayers; i++) {
            double[] inps=i==0?
                    inputs:layerOutputs[i-1];
            for (int j=0; j<neurons[i].length; j++) {

                neurons[i][j].proceed(inps);
                layerOutputs[i][j]=neurons[i][j].getOutput();
            }
        }

    }

    void formLayerOutputs() {
        layerOutputs = new double[neurons.length][];
        for (int i=0; i<neurons.length; i++) {
            int len = neurons[i].length;

            double[] outp =new double[len];

            layerOutputs[i]=outp;
        }
    }

    void formLayerErrors() {
        layerErrors = new double[neurons.length][];
        for (int i=0; i<neurons.length; i++) {
            int len = neurons[i].length;

            double[] err =new double[len];

            layerErrors[i]=err;
        }
    }

    public double[] getOutputs() {
        return this.getOutputs(kolLayers-1);
    }

    public double[] getOutputs(int layer) {
        return this.layerOutputs[layer];
    }

    public double[] getErrors() {
        return this.getErrors(kolLayers-1);
    }

    public double[] getErrors(int layer) {
        return this.layerErrors[layer];
    }

    public boolean train(double[][] inputSet,
                         double[][] outputSet,
                         double EPS,
                         double gamma,
                         int MAX_ITERS
    ) {

        double[][] errors = new double[kolLayers][];

        if (inputSet.length != outputSet.length) {
            throw new RuntimeException("Wrong training dimensions");
        }

        boolean success=false;
        int iter;

        for (iter=0; iter<MAX_ITERS; iter++) {
            double totalError=0;
           // System.out.println("---- Iteration "+iter);
            for (int smp=0; smp<inputSet.length; smp++) {
                double[] sample = inputSet[smp];
                double[] desiredOutputs = outputSet[smp];
                this.proceed(sample);

                //Backward propagation of errors

                for (int i=kolLayers-1; i>=0; i--) {
                    for (int j=0; j<neurons[i].length; j++) {
                        //Calculating errors
                        double delta;
                        double out = layerOutputs[i][j];
                        if (i==kolLayers-1) {
                            delta = desiredOutputs[j]-out;

                            layerErrors[kolLayers-1][j]=delta*out*(1.-out);

                        }
                        else {

                            delta=0.;
                            for (int dd = 0; dd < neurons[i + 1].length; dd++) {
                                delta += layerErrors[i + 1][dd] * neurons[i + 1][dd].getWeight(j);
                            }
                        }
                        double mult = (1.-out)*out;
                        delta*=mult;
                        layerErrors[i][j]=delta;
                        //Changing weights
                        for (int k=0; k<=neurons[i][j].getInputCount(); k++){
                            double inpt = i==0?
                                    Neuron.expandWithOne(sample)[k]:
                                    k==neurons[i][j].getInputCount()?1.:
                                    layerOutputs[i-1][k];
                            Neuron neuron = neurons[i][j];
                            double oldValue=neuron.getWeight(k);
                            double newValue=oldValue+gamma*delta*inpt;
                            neuron.setWeight(k, newValue);
                        }

                        //End of manipulations with layer
                    }
                }

                //End of backward propagation

                //End of current loop by samples
                totalError+=this.calculateSquareError();
            }

            if (totalError<EPS) {
                success=true;
                break;
            }

            //End of training loop
        }
        System.out.println("There were "+iter+" iterations");
        return success;
    }

    public Neuron[][] getNeurons() {
        return neurons;
    }

    public void setNeurons(Neuron[][] neurons) {
        this.neurons = neurons;
    }

    public double calculateSquareError() {
        double s=0;
        for (int j = 0; j < layerErrors[kolLayers-1].length; j++) {
                s+=layerErrors[kolLayers-1][j]*layerErrors[kolLayers-1][j];

            }

        return s;
    }

    public static BackPropagationNetwork load(String fileName) {
        BackPropagationNetwork network = null;
        Neuron[][] neurons=null;
        try(FileReader fr = new FileReader(fileName);
                BufferedReader br =new BufferedReader(fr)) {
            int kolInputs=Integer.parseInt(br.readLine());
            int kolLayers = Integer.parseInt(br.readLine());
            //Amount of neurons on layers
            String st = br.readLine();
            String[] kolStr=st.split(";");
            int[] kolNeuronsOnLayer = new int[kolLayers];
            for (int i=0; i<kolLayers; i++) {
                kolNeuronsOnLayer[i]=Integer.parseInt(kolStr[i]);
            }
            //Activation functions
            String stActiv = br.readLine();
            String[] functions=stActiv.split(";");
            //Creating instance
            network = new BackPropagationNetwork(
                    kolInputs,
                    functions,
                    kolNeuronsOnLayer
            );
            //Loading weights
            neurons = new Neuron[kolLayers][];
            for (int i = 0; i < kolLayers; i++) {
                neurons[i]=new Neuron[kolNeuronsOnLayer[i]];

                for (int j=0; j<neurons[i].length; j++) {

                    String wSt = br.readLine();
                    String[] wStArr = wSt.split(";");
                    double[] ws = new double[wStArr.length];

                    for (int k=0; k<wStArr.length; k++) {
                        ws[k]=Double.parseDouble(wStArr[k].trim().replace(',','.'));
                    }
                    neurons[i][j]=new Neuron(
                            loadFunctionByName(functions[i]),
                            ws,
                            ws.length
                            );
                    }

                }

        } catch (IOException e) {
            e.printStackTrace();
        }
        network.setNeurons(neurons);
        return network;
    }

    public void save(String fileName) {
        try(PrintWriter pw =new PrintWriter(fileName)) {
            pw.println(kolInputs);
            pw.println(kolLayers);
            for (int i=0; i<kolLayers; i++) {
                int kol=neurons[i].length;
                pw.print(kol);
                if (i!=kolLayers-1) pw.print(";");
            }
            pw.println();
            for (int i=0; i<kolLayers; i++) {
                String func=activFunctions[i].inform();
                pw.print(func);
                if (i!=kolLayers-1) pw.print(";");
            }
            pw.println();
            for (int i = 0; i < kolLayers; i++) {
                for (int j=0; j<neurons[i].length; j++) {
                    double[] w = neurons[i][j].getWeights();
                    for (int k=0; k<w.length; k++) {
                        pw.printf("%10.4f",w[k]);
                        if (k!=w.length-1) pw.print(";");
                    }
                    pw.println();
                }

            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static ActivationFunction loadFunctionByName(String s) {
        switch (s) {
            case "sigmoid": return new SigmoidFunction();
            case "treshold": return new TresholdFunction();
            default: return new SigmoidFunction();
        }
    }

}
