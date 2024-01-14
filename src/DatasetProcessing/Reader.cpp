#include "Utility.h"
#include "Reader.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

namespace parser {
	void read_image_file(std::vector<std::vector<uint8_t>>& images, const std::string& path, std::size_t limit) {
		auto buffer = read(path, 2051);

		if (!buffer) {
			return;
		}

		auto count   = read_header(buffer, 1);
		auto rows    = read_header(buffer, 2);
		auto columns = read_header(buffer, 3);

		//Skip the header
		//Cast to unsigned char is necessary cause signedness of char is
		//platform-specific
		auto image_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 16);

		if (limit > 0 && count > limit) {
			count = static_cast<unsigned int>(limit);
		}

		images.reserve(count);

		for (size_t i = 0; i < count; ++i) {
			images.push_back(std::vector<uint8_t>(1 * 28 * 28));

			for (size_t j = 0; j < rows * columns; ++j) {
				auto pixel   = *image_buffer++;
				images[i][j] = static_cast<uint8_t>(pixel);
			}
		}
	}

	void read_label_file(std::vector<uint8_t>& labels, const std::string& path, std::size_t limit) {
		auto buffer = read(path, 2049);

		if (!buffer) {
			return;
		}

		auto count = read_header(buffer, 1);

		//Skip the header
		//Cast to unsigned char is necessary cause signedness of char is
		//platform-specific
		auto label_buffer = reinterpret_cast<unsigned char*>(buffer.get() + 8);

		if (limit > 0 && count > limit) {
			count = static_cast<unsigned int>(limit);
		}

		labels.resize(count);

		for (size_t i = 0; i < count; ++i) {
			auto label = *label_buffer++;
			labels[i]  = static_cast<uint8_t>(label);
		}
	}

	std::vector<std::vector<uint8_t>> read_training_images(const std::string& Folder, std::size_t limit) {
		std::vector<std::vector<uint8_t>> images;
		read_image_file(images, Folder + "/train-images.idx3-ubyte", limit);
		return images;
	}

	std::vector<std::vector<uint8_t>> read_test_images(const std::string& Folder, std::size_t limit) {
		std::vector<std::vector<uint8_t>> images;
		read_image_file(images, Folder + "/t10k-images.idx3-ubyte", limit);
		return images;
	}

	std::vector<uint8_t> read_training_labels(const std::string& Folder, std::size_t limit) {
		std::vector<uint8_t> labels;
		read_label_file(labels, Folder + "/train-labels.idx1-ubyte", limit);
		return labels;
	}

	std::vector<uint8_t> read_test_labels(const std::string& Folder, std::size_t limit) {
		std::vector<uint8_t> labels;
		read_label_file(labels, Folder + "/t10k-labels.idx1-ubyte", limit);
		return labels;
	}

	dataset import_datasets(const std::string& Folder, std::size_t training_limit, std::size_t test_limit) {
		dataset dataset;

		dataset.training_images = read_training_images(Folder, training_limit);
		dataset.training_labels = read_training_labels(Folder, training_limit);

		dataset.test_images = read_test_images(Folder, test_limit);
		dataset.test_labels = read_test_labels(Folder, test_limit);

		return dataset;
	}
}
