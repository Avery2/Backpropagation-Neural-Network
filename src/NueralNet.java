

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import org.jblas.DoubleMatrix;

/**
 * @author averychan
 *
 */
public class NueralNet implements Serializable {

  private static final long serialVersionUID = 1L;
  private final static String NUERAL_NET_PATH =
      "/Users/averychan/eclipse-workspace/Backpropagation Neural Network/src/mynueralnet.ser";
  private final int numLayers, inputSize;
  private int[] shape;
  private DoubleMatrix[] weights, biases, activations, weightGradient, biasGradient;
  private double[] expected;
  double[] activationPartialsTemp;
  int[][] weight_shapes;

  /**
   * Constructor to make a new nueral network of the size specified by int[] shape and randomizes
   * weights and biases
   * 
   * @param layerSizes
   */
  public NueralNet(int[] layerSizes) {
    // initialize variables
    this.shape = layerSizes;
    numLayers = layerSizes.length;
    inputSize = layerSizes[0];

    // Set weight shapes
    weight_shapes = new int[numLayers - 1][2];
    for (int i = 0; i < numLayers - 1; i++) {
      weight_shapes[i][0] = layerSizes[i + 1];
      weight_shapes[i][1] = layerSizes[i];
      // System.out.println(Arrays.toString(weight_shapes[i]));
    }

    // Set weight matricies
    this.weights = new DoubleMatrix[weight_shapes.length];
    for (int i = 0; i < weight_shapes.length; i++) {
      weights[i] = DoubleMatrix.randn(weight_shapes[i][0], weight_shapes[i][1]);
    }

    // Set bias matricies
    this.biases = new DoubleMatrix[numLayers - 1];
    for (int i = 0; i < numLayers - 1; i++) {
      biases[i] = DoubleMatrix.zeros(layerSizes[i + 1]);
    }

    // Set blank activation matrix
    this.activations = new DoubleMatrix[numLayers];
    for (int i = 0; i < numLayers; i++) {
      activations[i] = new DoubleMatrix(layerSizes[i]);
    }
  }

  /**
   * Feeds forward one set of inputs a
   * 
   * @param activation The inputs of the nueral net
   * @return
   */
  public DoubleMatrix predict(double[] activation) {
    activations[0] = new DoubleMatrix(activation);

    // iterate through all layers
    for (int i = 0; i < numLayers - 1; i++) {
      activations[i + 1] = sigmoidElementWise(weights[i].mmul(activations[i]).add(biases[i]));
    }

    return activations[inputSize - 1];
  }

  public void backprop(int ans) {
    // initialize
    weightGradient = new DoubleMatrix[weights.length];
    for (int i = 0; i < weights.length; i++) {
      weightGradient[i] = weights[i].dup();
    }

    expected = new double[10];
    for (int i = 0; i < expected.length; i++) {
      if (i != ans) {
        expected[i] = 0;
      } else {
        expected[i] = 1;
      }
    }

    // calc partial activation for each layer
    double[][] activationPartials = new double[numLayers][];
    double foo = 0;

    // layer L
    activationPartialsTemp = new double[shape[numLayers - 1]];
    for (int i = 0; i < shape[numLayers - 1]; i++) {
      activationPartialsTemp[i] = 2 * (activations[inputSize - 1].get(i) - expected[i]);
    }

    activationPartials[3] = activationPartialsTemp;

    // Layers not including Layer L

    for (int i = 1; i < inputSize; i++) {
      activationPartialsTemp = new double[shape[numLayers - 1 - i]];
      for (int j = 0; j < shape[numLayers - 1 - i]; j++) {
        for (int n = 0; n < shape[numLayers - i]; n++) {
          foo += activationPartials[activationPartials.length - i][n]
              * activations[inputSize - i].get(n) * (1 - activations[inputSize - i].get(n))
              * weights[weights.length - i].get(j, n);
        }
        activationPartialsTemp[j] = foo;
        foo = 0;
      }
      activationPartials[activationPartials.length - 1 - i] = activationPartialsTemp;
    }

    // calc weight partials
    DoubleMatrix temp_dm_par = new DoubleMatrix();
    foo = 0;

    for (int i = 1; i < weights.length + 1; i++) {
      temp_dm_par.copy(weights[weights.length - i]);
      for (int j = 0; j < temp_dm_par.rows; j++) {
        for (int k = 0; k < temp_dm_par.columns; k++) {
          foo = activations[inputSize - 1 - i].get(k) * activations[inputSize - i].get(j)
              * (1 - activations[inputSize - i].get(j))
              * activationPartials[activationPartials.length - i][j];
          temp_dm_par.put(j, k, foo);
        }
      }
      weightGradient[weights.length - i].copy(temp_dm_par);
    }

    // Calc bias partials

    biasGradient = new DoubleMatrix[biases.length];
    for (int i = 0; i < biases.length; i++) {
      biasGradient[i] = biases[i].dup();
    }

    temp_dm_par = new DoubleMatrix();
    foo = 0;

    for (int i = 1; i < biases.length + 1; i++) {
      temp_dm_par.copy(biases[biases.length - i]);
      for (int j = 0; j < temp_dm_par.rows; j++) {
        for (int k = 0; k < temp_dm_par.columns; k++) {
          foo = activations[inputSize - i].get(j) * (1 - activations[inputSize - i].get(j))
              * activationPartials[activationPartials.length - i][j];
          temp_dm_par.put(j, k, foo);
        }
      }
      biasGradient[biases.length - i].copy(temp_dm_par);
    }

    System.out.println(weightGradient[0].toString());
  }

  public void applyBackprop(DoubleMatrix[] weightGradient, DoubleMatrix[] biasGradient) {
    for (int i = 0; i < weights.length; i++) {
      weights[i] = weights[i].add(weightGradient[i].neg());
    }
    for (int i = 0; i < biases.length; i++) {
      biases[i] = biases[i].add(biasGradient[i].neg());
    }
  }

  public void applyBackprop() {
    for (int i = 0; i < weights.length; i++) {
      weights[i] = weights[i].add(weightGradient[i].neg());
    }
    for (int i = 0; i < biases.length; i++) {
      biases[i] = biases[i].add(biasGradient[i].neg());
    }
  }

  /**
   * Saves the current state of the instance into a serialized file
   * 
   * @throws IOException
   */
  public void saveState() throws IOException {
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(NUERAL_NET_PATH));
    oos.writeObject(this);
    oos.close();
  }

  /**
   * Loads previous state of a NueralNetwork object from predefined serialized file
   * 
   * @return the saved NueralNet
   * @throws IOException
   * @throws ClassNotFoundException
   */
  public static NueralNet loadState() throws IOException, ClassNotFoundException {
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(NUERAL_NET_PATH));
    NueralNet nn = (NueralNet) ois.readObject();
    ois.close();
    return nn;
  }

  /**
   * @return
   */
  public DoubleMatrix[] getWeightGradient() {
    return weightGradient;
  }

  /**
   * @return
   */
  public DoubleMatrix[] getBiasGradient() {
    return biasGradient;
  }

  /**
   * @return
   */
  public DoubleMatrix[] getWeights() {
    return weights;
  }

  /**
   * @return
   */
  public DoubleMatrix[] getBiases() {
    return biases;
  }

  /**
   * @param weights
   */
  public void setWeights(DoubleMatrix[] weights) {
    this.weights = weights;
  }

  /**
   * @param biases
   */
  public void setBiases(DoubleMatrix[] biases) {
    this.biases = biases;
  }

  /**
   * @param activations
   */
  public void setActivations(DoubleMatrix[] activations) {
    this.activations = activations;
  }

  /**
   * Element-wise sigmoid operation for DoubleMatricies
   * 
   * @param layer Layer of nueral network
   * @return the layer with values squished by the sigmoid function
   */
  private static DoubleMatrix sigmoidElementWise(DoubleMatrix layer) {
    for (int i = 0; i < layer.length; i++) {
      layer.put(i, sigmoid(layer.get(i)));
    }
    return layer;
  }

  /**
   * applies sigmoid function to a value
   * 
   * @param valuevalue to be squished
   * @return squished value
   */
  private static double sigmoid(double value) {
    return (1 / (1 + Math.pow(Math.E, (-1 * value))));
  }
}
