#include "DatasetProcessing/Reader.h"
#include "Neural/Network.h"

#include <iostream>
//TODO: Display drawn version of image and label in a window and allow for going
//through them afterwards. Maybe use shaders to color a square based on the brightness.
int main(){
	//I compile to ../build/${PROJECT_NAME}
	auto dataset = parser::import_datasets("../../MNIST");

	auto ImagesTrain = dataset.training_images;
	auto ImagesTest = dataset.test_images;
	auto LabelsTrain = dataset.training_labels;
	auto LabelsTest = dataset.test_labels;

	//Labels contains a uint8_t for each value.
	//Images contains a vector<uint8_t> for each value.
	std::cout << "       | Training set | Test set |"<< "\n";
	std::cout << "-------|--------------|----------|" << "\n";
	std::cout << "Images |    " << ImagesTrain.size() << "     |   " << ImagesTest.size() << "  |\n";
	std::cout << "Labels |    " << LabelsTrain.size() << "     |   " << LabelsTest.size() << "  |\n\n";

	//Trains the Neural Network with the training data.
	arma::mat W1, b1, W2, b2;
	std::vector Temp = NN::Train(ImagesTrain, LabelsTrain, 2000, 0.2, 100, false);
	W1 = Temp[0];
	b1 = Temp[1];
	W2 = Temp[2];
	b2 = Temp[3];

	//Tests the model against the test data.
	NN::TestModel(ImagesTest, LabelsTest, W1, b1, W2, b2);

	return 0;
}
