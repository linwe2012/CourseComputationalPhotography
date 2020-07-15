#pragma once
#include <stdlib.h>
#include <tuple>

#include "utils.h"

// Get witdth & height from argv
inline auto getWH(char** argv, int a, int b) {
	double ia = atof(argv[a]);
	double ib = atof(argv[b]);
	return std::make_tuple(
		static_cast<int>(ia),
		static_cast<int>(ib),
		static_cast<float>(ia),
		static_cast<float>(ib)
	);
}



// guass(x) = a * exp(x^2/b)
inline constexpr auto guassianHelper(float sigma) {
	constexpr float sqrt_2pi = static_cast<float>(Math::sqrt(2.0 * Math::PI));
	float a = 1.0f / (sigma * sqrt_2pi);
	float b = 2 * sigma * sigma;
	return std::make_tuple(a, b);
}



inline auto initArgs(int argc, char*** pargv, std::string name, const char* args, int num_args) {
	char** argv = *pargv;
#ifndef _DEBUG
	if (argc < num_args) {
		std::cout << "Usage: " << name << " " << args << std::endl;
		return -1;
	}
#else
	static std::string of = "results/" + name + ".png";
	static const char* c[] = {
		".", "Lena.bmp", of.c_str(),
		"19", "19"
	};
	if (name == "GaussianFilter") {
		c[3] = "10";
	}
	if (name == "MedianFilter") {
		c[1] = "Lena_sp.png";
		c[3] = c[4] = "7";
	}
	
	*pargv = const_cast<char**>(c);
#endif // !_DEBUG
	return 0;
}
