#include "Network.h"
#include <cstddef>
#include <iostream>
#include <vector>

namespace NN {
	//Reformats the vectors into armadillo matrices.
	std::vector<arma::mat> Reformat(std::vector<std::vector<uint8_t>> Images, std::vector<uint8_t> Labels) {
		arma::mat LabelMatrix = arma::conv_to<arma::mat>::from(Labels).t();
		arma::mat ImageMatrix = arma::mat(Images[0].size(), Images.size());
		//Process each pixel of the image.
		for (int i=0;i<Images.size();i++) {
			for (int j=0;j<Images[i].size();j++) {
				ImageMatrix(j,i) = (int)Images[i][j];
			}
		}
		ImageMatrix = ImageMatrix / 255;
		return {ImageMatrix, LabelMatrix};
	}

	std::vector<arma::mat> Train(std::vector<std::vector<uint8_t>> Images, std::vector<uint8_t> Labels, size_t Iterations, double Alpha, int CheckpointInterval, bool SaveToFile) {
		//Initialize the weights and biases as random values close to 0.
		/*arma::mat W1 = arma::mat(10, 784, arma::fill::randu) * 0.05;
		arma::mat b1 = arma::mat(10, 1, arma::fill::randu) * 0.05;
		arma::mat W2 = arma::mat(10, 10, arma::fill::randu) * 0.05;
		arma::mat b2 = arma::mat(10, 1, arma::fill::randu) * 0.05;*/

		arma::mat W1 = arma::randn(10, 784) * 0.05;
		arma::mat b1 = arma::randn(10,1) * 0.05;
		arma::mat W2 = arma::randn(10,10) * 0.05;
		arma::mat b2 = arma::randn(10,1) * 0.05;

		//Convert the vectors to armadillo matrices.
		std::vector Temp = Reformat(Images, Labels);
		arma::mat ImageMatrix = Temp[0];
		arma::Mat<int> LabelMatrix = arma::conv_to<arma::Mat<int>>::from(Temp[1]);

		//Test: Prints an image
		/*arma::mat test = arma::mat(28,28);
		for (int i=0; i<28;i++) {
			for (int j=0; j<28; j++) {
				test(j,i) = ImageMatrix(i*28+j,1);
				std::cout << ceil(test(j,i)) << " ";
				if (j==27) {
					std::cout << "\n";
				}
			}
		}*/

		//Define variables
		arma::mat Z1, A1, Z2, A2;
		arma::mat dW1, db1, dW2, db2;

		//Get OneHot
		arma::Mat<int> OneHotY = OneHot(LabelMatrix);

		//Start main loop
		for (int i=0; i <= Iterations; i++) {
			//Don't know how to return multiple values and can't think of a better
			//way.
			//std::cout << "Forward" << std::endl;
			std::vector Forward = ForwardPropagate(W1, b1, W2, b2, ImageMatrix);
			Z1 = Forward[0];
			A1 = Forward[1];
			Z2 = Forward[2];
			A2 = Forward[3];
			//Z1.brief_print();
			//A1.brief_print();
			//Z2.brief_print();
			//A2.brief_print();
			//std::cout << "Backward" << std::endl;
			std::vector Backward = BackwardsPropagate(Z1, A1, Z2, A2, W2, ImageMatrix, LabelMatrix, OneHotY);
			dW1 = Backward[0];
			db1 = Backward[1];
			dW2 = Backward[2];
			db2 = Backward[3];
			//dW1.brief_print();
			//db1.brief_print();
			//dW2.brief_print();
			//db2.brief_print();

			//std::cout << "Update" << std::endl;
			std::vector Update = UpdateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, Alpha);
			W1 = Update[0];
			b1 = Update[1];
			W2 = Update[2];
			b2 = Update[3];
			if (i % (CheckpointInterval) == 0) {
				std::cout << "Iteration: " << i << "\n";
				std::cout << "Accuracy: " << GetAccuracy(GetPredictions(A2), LabelMatrix) * 100 << " %\n\n";
				if (SaveToFile) {
					Save(W1, b1, W2, b2);
				}
			}
		}
		return {W1, b1, W2, b2};
	};

	//Tests a model against test data.
	void TestModel(std::vector<std::vector<uint8_t>> Images, std::vector<uint8_t> Labels, arma::mat W1, arma::mat b1, arma::mat W2, arma::mat b2) {
		std::vector Temp = Reformat(Images, Labels);
		arma::mat ImageMatrix = Temp[0];
		arma::Mat<int> LabelMatrix = arma::conv_to<arma::Mat<int>>::from(Temp[1]);
		arma::mat A2 = ForwardPropagate(W1, b1, W2, b2, ImageMatrix)[3];
		std::cout << "Testdata Accuracy: " << GetAccuracy(GetPredictions(A2), LabelMatrix) * 100 << " %\n\n";
	}

	//Saves the weights and biases to binary files.
	void Save(arma::mat W1, arma::mat b1, arma::mat W2, arma::mat b2) {
		bool Success;
		std::cout << "Saving W1: ";
		Success = W1.save("W1.bin");
		if (Success) std::cout << "Success" << "\n"; else std::cout << "Failure" << "\n";

		std::cout << "Saving b1: ";
		Success = b1.save("b1.bin");
		if (Success) std::cout << "Success" << "\n"; else std::cout << "Failure" << "\n";

		std::cout << "Saving W2: ";
		Success = W2.save("W2.bin");
		if (Success) std::cout << "Success" << "\n"; else std::cout << "Failure" << "\n";

		std::cout << "Saving b2: ";
		Success = b2.save("b2.bin");
		if (Success) std::cout << "Success" << "\n\n"; else std::cout << "Failure" << "\n\n";
	};

	//
	std::vector<arma::mat> ForwardPropagate(arma::mat W1, arma::mat b1, arma::mat W2, arma::mat b2, arma::mat ImageMatrix) {
		//std::cout << "Z1" << std::endl;
		arma::mat Z1 = W1 * ImageMatrix + arma::repmat(b1, 1, ImageMatrix.n_cols);
		//W1.brief_print();
		//ImageMatrix.brief_print();
		//(W1 * ImageMatrix).brief_print();
		//b1.brief_print();
		//Z1.brief_print();
		//std::cout << "A1" << std::endl;
		arma::mat A1 = ReLU(Z1);
		//std::cout << "Z2" << std::endl;
		arma::mat Z2 = W2 * A1 + arma::repmat(b2, 1, ImageMatrix.n_cols);
		//std::cout << "A2" << std::endl;
		arma::mat A2 = SoftMax(Z2);
		//std::cout << "Forward done" << std::endl;
		return {Z1, A1, Z2, A2};
	};

	arma::mat ReLU(arma::mat Z) {
		Z.for_each([](arma::mat::elem_type& val){if(val<0.0) val = 0.0;});
		return Z;
	};

	arma::mat SoftMax(arma::mat Z) {
		arma::mat z = arma::repmat(arma::sum(arma::exp(Z)), 10, 1);
		return arma::exp(Z) / z;
	};

	//
	std::vector<arma::mat> BackwardsPropagate(arma::mat Z1, arma::mat A1, arma::mat Z2, arma::mat A2, arma::mat W2, arma::mat Images, arma::Mat<int> Labels, arma::Mat<int> OneHotY) {
		double LS = Labels.size();
		//std::cout << "dZ2" << std::endl;
		arma::mat dZ2 = 2*(A2 - OneHotY);
		//std::cout << "dW2" << std::endl;
		arma::mat dW2 = (1.0/LS) * dZ2 * A1.t();
		//std::cout << "db2" << std::endl;
		arma::mat db2 = (1 / LS) * arma::sum(dZ2, 1);
		//std::cout << "dZ1" << std::endl;
		arma::mat dZ1 = (W2.t() * dZ2) % DerivativeReLU(Z1);
		//std::cout << "dW1" << std::endl;
		arma::mat dW1 = (1.0/LS) * dZ1 * Images.t();
		//std::cout << "db1" << std::endl;
		arma::mat db1 = (1.0/LS) * arma::sum(dZ1, 1);
		//std::cout << "Back done" << std::endl;
		return {dW1, db1, dW2, db2};
	};

	arma::Mat<int> OneHot(arma::Mat<int> Labels) {
		//Create a matrix with rows equal to the number of different numbers,
		//assuming they start at 0, and columns equal to the number of labels.
		arma::Mat<int> OneHotY = arma::Mat<int>(Labels.max() + 1, Labels.size());
		//Goes through all labels and assings the label to the current
		//column's nth row.
		for (int i=0; i < Labels.size(); i++) {
			OneHotY(Labels[i],i) = 1;
		}
		return OneHotY;
	};

	arma::mat DerivativeReLU(arma::mat Z) {
		return Z.for_each([](arma::mat::elem_type& val){val = (val>0);});
	};

	//Updates the weights and biases.
	std::vector<arma::mat> UpdateParams(arma::mat W1, arma::mat b1, arma::mat W2, arma::mat b2, arma::mat dW1, arma::mat db1, arma::mat dW2, arma::mat db2, double alpha) {
		//std::cout << "W1" << std::endl;
		W1 -= alpha * dW1;
		//std::cout << "b1" << std::endl;
		b1 -= alpha * db1;
		//std::cout << "W2" << std::endl;
		W2 -= alpha * dW2;
		//std::cout << "b2" << std::endl;
		b2 -= alpha * db2;
		//std::cout << "Updated" << std::endl;
		return {W1, b1, W2, b2};
	};

	//Collapses the matrix into 1xn_colums.
	arma::Mat<arma::uword> GetPredictions(arma::mat A2) {
		return arma::index_max(A2, 0);
	};

	//Compares the Predictions and the Labels and counts their amount. Dividing by
	//the amount of Labels returns the percentage as a decimal.
	double GetAccuracy(arma::umat Predictions, arma::Mat<int> Labels) {
		double Correct = arma::as_scalar(arma::sum(Predictions == Labels, 1));
		return  Correct / Labels.size();
	}
}
