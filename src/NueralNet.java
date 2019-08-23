import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.Arrays;

import org.jblas.DoubleMatrix;

public class NueralNet implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int[] shape;
	DoubleMatrix[] weights, biases, activations;

	/**
	 * Constructor to make a new nueral network of the size specified by int[] shape and randomizes weights and biases
	 * 
	 * @param shape
	 */
	public NueralNet(int[] shape) {
		this.shape = shape;

		// Set weight shapes
		int[][] weight_shapes = new int[shape.length - 1][2];
		for (int i = 0; i < shape.length - 1; i++) {
			weight_shapes[i][0] = shape[i + 1];
			weight_shapes[i][1] = shape[i];
		}

		// Set weight matricies
		this.weights = new DoubleMatrix[weight_shapes.length];
		for (int i = 0; i < weight_shapes.length; i++) {
			weights[i] = DoubleMatrix.randn(weight_shapes[i][0], weight_shapes[i][1]);
		}

		// Set bias matricies
		this.biases = new DoubleMatrix[shape.length - 1];
		for (int i = 0; i < shape.length - 1; i++) {
			biases[i] = DoubleMatrix.zeros(shape[i + 1]);
		}
	}

	/**
	 * Returns the output DoubleMatrix for input vector a
	 * 
	 * @param a
	 * @return
	 */
	public DoubleMatrix predict(double[] a) {
		// Set activation matricies
		this.activations = new DoubleMatrix[shape.length];
		for (int i = 0; i < activations.length; i++) {
			activations[i] = new DoubleMatrix(shape[i]);
		}

		activations[0] = new DoubleMatrix(a);

		// iterate through all layers
		for (int i = 0; i < shape.length - 1; i++) {
			activations[i + 1] = sigmoid_ew(weights[i].mmul(activations[i]).add(biases[i]));
		}

		return activations[activations.length - 1];
	}

	public static double cost(DoubleMatrix out, int label) {
		double cost = 0;
		for (int i = 0; i < out.length; i++) {
			if (i != label)
				cost += (Math.pow(out.get(i), 2));
			else
				cost += Math.pow((out.get(i) - 1), 2);
		}
		return cost;
	}

	public void backprop(int ans) {
		DoubleMatrix[] w_grad, b_grad;
		w_grad = new DoubleMatrix[weights.length];

		for (int i = 0; i < weights.length; i++) {
			w_grad[i] = weights[i].dup();
		}

		// setup expected
		double[] expected = new double[10];
		for (int i = 0; i < expected.length; i++) {
			if (i != ans) {
				expected[i] = 0;
			} else {
				expected[i] = 1;
			}
		}

		System.out.println(activations[3].rows + " " + activations[3].columns);

		// Calc last layer
		double v = 0;
		for (int j = 0; j < weights[2].rows; j++) {
			for (int k = 0; k < weights[2].columns; k++) {
				v = 2 * (activations[3].get(j) - expected[j]) * activations[3].get(j) * (1 - activations[3].get(j))
						* activations[2].get(k);
				w_grad[2].put(j, k, v);
			}
		}
		System.out.println(weights[2].toString());
		System.out.println(w_grad[2].toString());
	}
	
	public void saveMe() throws IOException {
		FileOutputStream fos = new FileOutputStream(
				"/Users/averychan/eclipse-workspace/Backpropagation Nueral Network/src/mynueralnet.ser");
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(this);
		oos.close();
		System.out.println("Object saved.");
	}

	public static NueralNet load_nn() throws IOException, ClassNotFoundException {
		FileInputStream fis;
		fis = new FileInputStream(
				"/Users/averychan/eclipse-workspace/Backpropagation Nueral Network/src/mynueralnet.ser");
		ObjectInputStream ois = new ObjectInputStream(fis);
		ois.close();
		return (NueralNet) ois.readObject();
	}

	/**
	 * Element-wise sigmoid operation for DoubleMatricies
	 * 
	 * @param dm
	 * @return
	 */
	private static DoubleMatrix sigmoid_ew(DoubleMatrix dm) {
		for (int i = 0; i < dm.length; i++) {
			dm.put(i, sigmoid(dm.get(i)));
		}
		return dm;
	}

	/*
	 * applies sigmoid function to a value
	 * 
	 * @param in
	 * 
	 * @return squished value
	 */
	private static double sigmoid(double in) {
		return (1 / (1 + Math.pow(Math.E, (-1 * in))));
	}
}