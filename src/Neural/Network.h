#pragma once

#include <vector>
#include <armadillo>

namespace NN {
	std::vector<arma::mat> Reformat(std::vector<std::vector<uint8_t>> Images, std::vector<uint8_t> Labels);

	//Trains the NN on a set of Images with their respective labels. Returns a
	//vector of matrices W1, b1, W2 and b2.
	std::vector<arma::mat> Train(std::vector<std::vector<uint8_t>> TrainingImages, std::vector<uint8_t> TrainingLabels, size_t Iterations=100, double Alpha=0.1, int CheckpointInterval=10, bool SaveToFile=false);

	arma::mat ReLU(arma::mat Z);

	arma::mat DerivativeReLU(arma::mat Z);

	arma::mat SoftMax(arma::mat Z);

	std::vector<arma::mat> ForwardPropagate(arma::mat W1, arma::mat b1, arma::mat W2, arma::mat b2, arma::mat ImageMatrix);

	std::vector<arma::mat> BackwardsPropagate(arma::mat Z1, arma::mat A1, arma::mat Z2, arma::mat A2, arma::mat W2,arma::mat Images, arma::Mat<int> Labels, arma::Mat<int> OneHotY);

	arma::Mat<int> OneHot(arma::Mat<int> Labels);

	std::vector<arma::mat> UpdateParams(arma::mat W1, arma::mat b1, arma::mat W2, arma::mat b2, arma::mat dW1, arma::mat db1, arma::mat dW2, arma::mat db2, double alpha);

	//Saves the weights and biases to binary files.
	void Save(arma::mat W1, arma::mat b1, arma::mat W2, arma::mat b2);

	//Loads saved weights from a give folder.
	void Load();

	arma::Mat<arma::uword> GetPredictions(arma::mat A2);

	double GetAccuracy(arma::umat Predictions, arma::Mat<int> Labels);

	void TestModel(std::vector<std::vector<uint8_t>> TestImages, std::vector<uint8_t> TestLabels, arma::mat W1, arma::mat b1, arma::mat W2, arma::mat b2);
}
