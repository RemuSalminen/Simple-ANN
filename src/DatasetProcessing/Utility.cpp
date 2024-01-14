#include "Utility.h"

#include <fstream>
#include <iostream>
#include <cstdint>

namespace parser {

	std::unique_ptr<char[]> read(const std::string& path, uint32_t MagicNumber) {
		std::ifstream File;
		//No need for ouput. MNIST files are not in text. MNIST files are stored in
		//MSB first format.
		File.open(path, std::ios::in | std::ios::binary | std::ios::ate);

		//Safeguard
		if (!File) {
			std::cout << "No file in " << path << std::endl;
			return {};
		}

		//Gets the current position and in turn the size of the file.
		auto size = File.tellg();
		std::unique_ptr<char[]> buffer(new char[size]);

		//Go to the beginning of the file -> read data into buffer -> close file
		File.seekg(0);
		File.read(buffer.get(), size);
		File.close();

		//Make sure the Magic Numbers match
		uint32_t magic = read_header(buffer, 0);
		if (magic != MagicNumber) {
			std::cout << "Incorrect Magic Number!" << std::endl;
			return {};
		}

		//Check the file for possible corruption
		uint32_t count = read_header(buffer, 1);
		if (magic == 2051) {
			uint32_t rows = read_header(buffer, 2);
			uint32_t colums = read_header(buffer, 3);

			if (size < count*rows*colums+16) {
				std::cout << "The file is too small!" << std::endl;
				return {};
			}
		} else if (magic == 2049) {
			if (size < count+8) {
				std::cout << "The file is too small!" << std::endl;
				return {};
			}
		}
		return buffer;
	}

	//Reads the files header and extracts a value from a given position
	uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position) {
		auto header = reinterpret_cast<uint32_t*>(buffer.get());

		//Does some magic
		auto value = *(header + position);
		return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0x0000FF00) | (value >> 24);
	}
}
