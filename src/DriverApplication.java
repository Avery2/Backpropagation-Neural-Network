
import java.io.FileNotFoundException;
import java.io.IOException;
import org.jblas.DoubleMatrix;

/**
 * @author averychan
 *
 */
public class DriverApplication {

  static double[][] input;
  private static int step_size;
  private static DoubleMatrix[] weightGradientAverage;
  private static DoubleMatrix[] biasGradientAverage;

  public static void main(String[] args)
      throws FileNotFoundException, IOException, ClassNotFoundException {
    // Setup
    System.out.println("Starting neural network...");
    double[][] data = ReadData.getDigits();
    int[] labels = ReadData.getLabels();

    // new nueral net
    int[] layerSizes = {784, 16, 16, 10};
    NueralNet nn = new NueralNet(layerSizes);

    // load nueral net
    // NueralNet nn = NueralNet.load_nn(); // TODO Broken

    // predict 50000 examples and show accuracy
    testAll(nn, data, labels);

    nn.saveState();

    // back prop
    for (int i = 0; i < labels.length; i++) {
      // for (int i = 0; i < 50000; i++) {
      nn.predict(data[i]);
      nn.backprop(labels[i]);
      nn.applyBackprop();
      if (i % 10000 == 0)
        System.out.println(i);
    }

    // for (int i = 0; i < 500; i++) {
    // StochasticDecent(nn, data, labels, i);
    // System.out.print(i);
    // }

    // save nueral net
    // nn.saveMe();
    // System.out.println(nn.getWeights()[0].toString());

    testAll(nn, data, labels);

    System.out.println("\nEnd.");
  }

  public static void StochasticDecent(NueralNet nn, double[][] data, int[] labels, int num) {
    step_size = 100; // TODO is that what step-size is?
    weightGradientAverage = new DoubleMatrix[nn.getWeights().length];
    biasGradientAverage = new DoubleMatrix[nn.getBiases().length];

    // initialize weightGradientAverage
    for (int i = 0; i < nn.getWeights().length; i++) {
      weightGradientAverage[i] =
          DoubleMatrix.zeros(nn.getWeights()[i].rows, nn.getWeights()[i].columns);
    }

    // initialize biasGradientAverage
    for (int i = 0; i < nn.getBiases().length; i++) {
      biasGradientAverage[i] =
          DoubleMatrix.zeros(nn.getBiases()[i].rows, nn.getBiases()[i].columns);
    }

    for (int i = 0; i < step_size; i++) {
      nn.predict(data[num * 100 + i]);
      nn.backprop(labels[num * 100 + i]);
      for (int j = 0; j < nn.getWeights().length; j++) {
        weightGradientAverage[j] = weightGradientAverage[j].add(nn.getWeightGradient()[j].neg());
      }

      for (int j = 0; j < nn.getBiases().length; j++) {
        biasGradientAverage[j] = biasGradientAverage[j].add(nn.getBiasGradient()[j].neg());
      }
    }

    for (int i = 0; i < nn.getWeights().length; i++) {
      weightGradientAverage[i] = weightGradientAverage[i].div(step_size);
    }

    for (int i = 0; i < nn.getBiases().length; i++) {
      biasGradientAverage[i] = biasGradientAverage[i].div(step_size);
    }

    System.out.println("wg " + nn.getWeightGradient()[0].toString());
    System.out.println("g " + weightGradientAverage[0].toString());
    System.out.println("w " + nn.getWeights()[0].toString() + "\n");

    nn.applyBackprop(weightGradientAverage, biasGradientAverage);

  }

  private static void testAll(NueralNet nn, double[][] data, int[] labels) {
    System.out.print("\nFeeding forward all examples...");
    long start = System.nanoTime();

    boolean[] answers = new boolean[50000];
    for (int i = 0; i < 50000; i++) {
      answers[i] = nn.predict(data[i]).argmax() == labels[i];
    }
    System.out.println("\t" + (System.nanoTime() - start) / 1000000000.0 + " seconds.");
    int correct = 0;
    for (int i = 0; i < 50000; i++) {
      if (answers[i])
        correct++;
    }
    System.out.println("The network got " + correct + " examples correct, which is "
        + (double) correct / 500.0 + "% accuracy");
  }
}

// TODO Stochastic Decent
// TODO refactor
// TODO train
// TODO visualiztion
// TODO make own handwriter and apply to nn
