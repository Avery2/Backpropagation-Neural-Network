import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

import org.jblas.DoubleMatrix;

public class program {

	static double[][] input;

	public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
		// Setup
		System.out.println("Starting neural network...");
		double data[][] = ReadData.getDigits();
		int labels[] = ReadData.getLabels();

		// new nueral net
//		int[] layer_sizes = { 784, 16, 16, 10 };
//		NueralNet nn = new NueralNet(layer_sizes);

		// load nueral net
		NueralNet nn = NueralNet.load_nn();

		// predict 50000 examples and show accuracy
		testAll(nn, data, labels);

		// predict 1 example, show cost function
//		System.out.println("\nTesting one example...");
//		DoubleMatrix out_ex0 = nn.predict(data[0]); // TODO eventually you dont want doublematrix in this class; Only have it here to print out
//		System.out.println(out_ex0.toString());
//		System.out.println("Correct answer: " + labels[0]);
//		System.out.println("Predicted answer: " + out_ex0.argmax());
//		System.out.println("Cost: " + NueralNet.cost(out_ex0, labels[0]));
		
		nn.saveMe();

		// back prop
//		for (int i=0; i<labels.length; i++) {
		for (int i = 0; i < 50000; i++) {
			nn.predict(data[i]);
			nn.backprop(labels[i]);
			if (i % 10000 == 0)
				System.out.println(i);
		}
//		nn.backprop(labels[0]);

		// save nueral net
//		nn.saveMe();
		testAll(nn, data, labels);

//		System.out.println("Same? "+a.equals(nn.weights[0]));

		// end
		System.out.println("\nEnd.");
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
		System.out.println("The network got " + correct + " examples correct, which is " + (double) correct / 500.0
				+ "% accuracy");
	}
}

// TODO Structure such that you can train individuals?
// TODO backpropogation -> adjust values
// TODO refactor
// TODO train
// TODO visualiztion
// TODO make own handwriter and apply to nn