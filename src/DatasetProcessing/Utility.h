#pragma once
#include <memory>
#include <string>

namespace parser {

	//Reads the MNIST file and returns it as an array. Returns empty on error.
	std::unique_ptr<char[]> read(const std::string& path, uint32_t MagicNumber);

	//Reads the files header and extracts a value from a given position.
	uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position);
}
