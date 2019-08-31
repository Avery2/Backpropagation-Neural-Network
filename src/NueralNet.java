import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;

import org.jblas.DoubleMatrix;

public class NueralNet implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int[] shape;
	DoubleMatrix[] weights, biases, activations, w_grad, b_grad; // weights -> 2d, biases -> 1d, activations -> 1d

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
			System.out.println(Arrays.toString(weight_shapes[i]));
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
		
		// set blank activation matricy
		this.activations = new DoubleMatrix[shape.length];
		for (int i = 0; i < activations.length; i++) {
			activations[i] = new DoubleMatrix(shape[i]);
		}
	}

	/**
	 * Returns the output DoubleMatrix for input vector a
	 * 
	 * @param a
	 * @return
	 */
	public DoubleMatrix predict(double[] a) {
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
//		DoubleMatrix[] w_grad, b_grad;

		// create DoubleMatrix for weight gradients
		w_grad = new DoubleMatrix[weights.length];
		for (int i = 0; i < weights.length; i++) {
			w_grad[i] = weights[i].dup();
		}

		// setup expected values as double[]
		double[] expected = new double[10];
		for (int i = 0; i < expected.length; i++) {
			if (i != ans) {
				expected[i] = 0;
			} else {
				expected[i] = 1;
			}
		}

//		System.out.println("\nCalculating Activation Partials...");

//		System.out.println(Arrays.toString(shape));
//		System.out.println(Arrays.toString(activations));
		
		// calc partial a for each layer
		double[][] a_par = new double[shape.length][];

		// layer L
		double[] temp_ap = new double[shape[shape.length - 1]];
		for (int i = 0; i < shape[shape.length - 1]; i++) {
			temp_ap[i] = 2 * (activations[activations.length - 1].get(i) - expected[i]);
		}

		a_par[3] = temp_ap;
		
//		System.out.println(Arrays.toString(temp_ap));

//		System.out.println("a_par\n"+Arrays.deepToString(a_par));

//		System.out.println(activations[activations.length - 1].toString());
//		System.out.println(Arrays.toString(temp_l1));

		// layer L-1
		temp_ap = new double[shape[shape.length - 2]];
		double foo = 0;
		for (int j = 0; j < shape[shape.length - 2]; j++) {
			for (int n = 0; n < shape[shape.length - 1]; n++) {
				foo += a_par[3][n] * activations[activations.length - 1].get(n)
						* (1 - activations[activations.length - 1].get(n)) * weights[2].get(j, n);
			}
			temp_ap[j] = foo;
			foo = 0;
		}

		a_par[2] = temp_ap;

//		System.out.println("a_par\n"+Arrays.deepToString(a_par));

		// layer L-2
		temp_ap = new double[shape[shape.length - 3]];
		foo = 0;
		for (int j = 0; j < shape[shape.length - 3]; j++) {
			for (int n = 0; n < shape[shape.length - 2]; n++) {
				foo += a_par[2][n] * activations[activations.length - 2].get(n)
						* (1 - activations[activations.length - 2].get(n)) * weights[1].get(j, n);
			}
			temp_ap[j] = foo;
			foo = 0;
		}
		
		a_par[1] = temp_ap;
		
		// layer L-3
				temp_ap = new double[shape[shape.length - 4]];
				foo = 0;
				for (int j = 0; j < shape[shape.length - 4]; j++) {
					for (int n = 0; n < shape[shape.length - 3]; n++) {
						foo += a_par[1][n] * activations[activations.length - 3].get(n)
								* (1 - activations[activations.length - 3].get(n)) * weights[0].get(j, n);
					}
					temp_ap[j] = foo;
					foo = 0;
				}

		a_par[0] = temp_ap;

//		System.out.println("a_par\n"+Arrays.deepToString(a_par));
//		System.out.println(Arrays.toString((a_par[0])));
//		System.out.println(Arrays.toString((a_par[1])));
//		System.out.println(Arrays.toString((a_par[2])));

		// layer L-3
		temp_ap = new double[shape[shape.length - 4]];
		foo = 0;
		for (int j = 0; j < shape[shape.length - 4]; j++) {
			for (int n = 0; n < shape[shape.length - 3]; n++) {
				foo += a_par[2][n] * activations[activations.length - 3].get(n)
						* (1 - activations[activations.length - 3].get(n)) * weights[0].get(j, n);
			}
			temp_ap[j] = foo;
			foo = 0;
		}

		a_par[0] = temp_ap;

//		System.out.println("a_par\n"+Arrays.deepToString(a_par));

//		System.out.println("Done.\nCalculating Weight partials...");

		// calc weight partials
		DoubleMatrix temp_dm_par = new DoubleMatrix();
		foo = 0;

		// layer L
		temp_dm_par.copy(weights[weights.length - 1]);

		for (int j = 0; j < temp_dm_par.rows; j++) {
			for (int k = 0; k < temp_dm_par.columns; k++) {
				foo = activations[activations.length - 2].get(k) * activations[activations.length - 1].get(j)
						* (1 - activations[activations.length - 1].get(j)) * a_par[a_par.length - 1][j];
				temp_dm_par.put(j, k, foo);
			}
		}

		w_grad[2].copy(temp_dm_par);
//		System.out.println(weights[weights.length-1].length);
//		System.out.println(w_grad[0].length); // needs to be bigger?

		// layer L-1

		foo = 0;

		temp_dm_par.copy(weights[weights.length - 2]);
		for (int j = 0; j < temp_dm_par.rows; j++) {
			for (int k = 0; k < temp_dm_par.columns; k++) {
				foo = activations[activations.length - 3].get(k) * activations[activations.length - 2].get(j)
						* (1 - activations[activations.length - 2].get(j)) * a_par[a_par.length - 2][j];
				temp_dm_par.put(j, k, foo);
			}
		}

		w_grad[1].copy(temp_dm_par);

		// layer L-2

		foo = 0;

		temp_dm_par.copy(weights[weights.length - 3]);
		for (int j = 0; j < temp_dm_par.rows; j++) {
			for (int k = 0; k < temp_dm_par.columns; k++) {
				foo = activations[activations.length - 4].get(k) * activations[activations.length - 3].get(j)
						* (1 - activations[activations.length - 3].get(j)) * a_par[a_par.length - 3][j];
				temp_dm_par.put(j, k, foo);
			}
		}

		w_grad[0].copy(temp_dm_par);

//		System.out.println("Done.");
//		System.out.println(Arrays.deepToString(w_grad));
//		for (int i = 0; i < w_grad.length; i++) {
//			System.out.print(i + ": " + w_grad[i].length + " ");
//		}
//		System.out.println();

		// Clac bias partials

		b_grad = new DoubleMatrix[biases.length];
		for (int i = 0; i < biases.length; i++) {
			b_grad[i] = biases[i].dup();
		}

//		System.out.println("Calculating bias partials...");

		temp_dm_par = new DoubleMatrix();
		foo = 0;

		// layer L
		temp_dm_par.copy(biases[biases.length - 1]);

		for (int j = 0; j < temp_dm_par.rows; j++) {
			foo = activations[activations.length - 1].get(j) * (1 - activations[activations.length - 1].get(j))
					* a_par[a_par.length - 1][j];
			temp_dm_par.put(j, foo);
		}

		b_grad[2].copy(temp_dm_par);
//		System.out.println(biases[biases.length-1].length);
//		System.out.println(b_grad[0].length); // needs to be bigger?

		// layer L-1

		foo = 0;

		temp_dm_par.copy(biases[biases.length - 2]);
		for (int j = 0; j < temp_dm_par.rows; j++) {
			foo = activations[activations.length - 2].get(j) * (1 - activations[activations.length - 2].get(j))
					* a_par[a_par.length - 2][j];
			temp_dm_par.put(j, foo);
		}

		b_grad[1].copy(temp_dm_par);

		// layer L-2

		foo = 0;

		temp_dm_par.copy(biases[biases.length - 3]);
		for (int j = 0; j < temp_dm_par.rows; j++) {
			foo = activations[activations.length - 3].get(j) * (1 - activations[activations.length - 3].get(j))
					* a_par[a_par.length - 3][j];
			temp_dm_par.put(j, foo);
		}

		b_grad[0].copy(temp_dm_par);

//		System.out.println("Done.");

//		System.out.print("Applying gradient");

//		System.out.println(Arrays.deepToString(biases));
//		System.out.println(Arrays.deepToString(b_grad));
		
		for (int i = 0; i < weights.length; i++) {
			weights[i] = weights[i].add(w_grad[i].neg());
		}
		for (int i = 0; i < biases.length; i++) {
			biases[i] = biases[i].add(b_grad[i].neg());
		}
//		System.out.println("Done");
//		System.out.println(Arrays.deepToString(biases));

	}

	public void saveMe() throws IOException {
		FileOutputStream fos = new FileOutputStream(
				"/Users/averychan/eclipse-workspace/Backpropagation Neural Network/src/mynueralnet.ser");
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(this);
		oos.close();
		System.out.println("Object saved.");
	}

	public static NueralNet load_nn() throws IOException, ClassNotFoundException {
		FileInputStream fis;
		fis = new FileInputStream(
				"/Users/averychan/eclipse-workspace/Backpropagation Neural Network/src/mynueralnet.ser");
		ObjectInputStream ois = new ObjectInputStream(fis);
		NueralNet nn = (NueralNet) ois.readObject();
		ois.close();
		return nn;
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