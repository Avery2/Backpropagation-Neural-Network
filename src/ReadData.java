import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class ReadData {

  /**
   * Reads an unsigned 32 bit integer
   * 
   * @param input A FileInputStream from which to read from.
   * @return Returns the integer's value.
   * @throws IOException
   */
  private static int read32bitInt(BufferedInputStream input) throws IOException {
    int x = 0;
    for (int i = 3; i >= 0; i--) {
      x += input.read() * Math.pow(256, i);
    }
    return x;
  }

  /**
   * Returns 2d array with 50000 handwritten digits.
   * 
   * @param n The specifies which image you wish to load, 1=first image, 2=second image, n = nth
   *          image
   * @return double[] array of image's pixel values
   * @throws FileNotFoundException
   * @throws IOException
   */
  static double[][] getDigits() throws FileNotFoundException, IOException {
    // System.out.print("Getting data...");

    // long start = System.nanoTime();

    // Setup
    String fileName =
        "/Users/averychan/eclipse-workspace/Backpropagation Neural Network/src/training-set-images.bin";
    File file = new File(fileName);
    BufferedInputStream inputStream = new BufferedInputStream(new FileInputStream(file));
    double[][] input = new double[50000][784];

    // Read meta data
    read32bitInt(inputStream);
    read32bitInt(inputStream);
    read32bitInt(inputStream);
    read32bitInt(inputStream);

    // Read handwritten digits
    for (int i = 0; i < 50000; i++) {
      for (int j = 0; j < 784; j++) {
        // input[i][j] = sigmoid(inputStream.read()); // BAD
        input[i][j] = inputStream.read();

      }
    }

    // System.out.println("\t" + (System.nanoTime() - start) / 1000000000.0 + " seconds.");

    inputStream.close();
    return input;

  }

  /**
   * Gets the answers to the handwritten images.
   * 
   * @return Integer array that lists the answers to the handwritten numbers in order.
   * @throws IOException
   */
  static int[] getLabels() throws IOException {
    String fileName;
    File file;
    BufferedInputStream inputStream;

    fileName =
        "/Users/averychan/eclipse-workspace/Backpropagation Neural Network/src/training-set-labels.bin";
    file = new File(fileName);

    inputStream = new BufferedInputStream(new FileInputStream(file));
    read32bitInt(inputStream);
    int[] labels = new int[read32bitInt(inputStream)];
    for (int i = 0; i < labels.length; i++) {
      labels[i] = inputStream.read();
    }
    inputStream.close();
    return labels;
  }
}
