#pragma once
#include <vector>
#include <string>

namespace parser {
	struct dataset {
		std::vector<std::vector<uint8_t>> training_images; ///< The training images
		std::vector<std::vector<uint8_t>> test_images;     ///< The test images
		std::vector<uint8_t> training_labels; ///< The training labels
		std::vector<uint8_t> test_labels;     ///< The test labels

		/*!
		* \brief Resize the training set to new_size
		* If new_size is bigger than the current size, this function has no effect.
		* \param new_size The size to resize the training sets to.
		*/
		void resize_training(std::size_t new_size) {
			if (training_images.size() > new_size) {
				training_images.resize(new_size);
				training_labels.resize(new_size);
			}
		}

		/*!
		* \brief Resize the test set to new_size
		* If new_size is bigger than the current size, this function has no effect.
		* \param new_size The size to resize the test sets to.
		*/
		void resize_test(std::size_t new_size) {
			if (test_images.size() > new_size) {
				test_images.resize(new_size);
				test_labels.resize(new_size);
			}
		}
	};

	void read_image_file(std::vector<std::vector<uint8_t>>& images, const std::string& path, std::size_t limit);

	void read_label_file(std::vector<uint8_t>& labels, const std::string& path, std::size_t limit = 0);

	std::vector<std::vector<uint8_t>> read_training_images(const std::string& Folder, std::size_t limit);

	std::vector<std::vector<uint8_t>> read_test_images(const std::string& Folder, std::size_t limit);

	std::vector<uint8_t> read_training_labels(const std::string& Folder, std::size_t limit);

	std::vector<uint8_t> read_test_labels(const std::string& Folder, std::size_t limit);

	//Imports the dataset from the files to a struct of vectors.
	dataset import_datasets(const std::string& Folder, std::size_t training_limit = 0, std::size_t test_limit = 0);
}
