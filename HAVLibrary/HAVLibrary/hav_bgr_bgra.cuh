#pragma once
#include <algorithm>

void hav_nv12_bgra32_SDR(unsigned char* cuLuma,
	unsigned char* cuChroma,
	unsigned int width,
	unsigned int heigth,
	unsigned char* destImage,
	unsigned int alpha = 255,
	bool inverted = false
);

void hav_p016_HDR10_bgr32_SDR(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int p016Width,
	unsigned int p016Height,
	unsigned char* rgbImage,
	bool inverted = false
);
void hav_p016_HDR10_bgr64_HDR10(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int p016Width,
	unsigned int p016Height,
	unsigned short* rgbImage,
	bool inverted = false
);
void hav_p016_HDR10_bgra32_SDR(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int p016Width,
	unsigned int p016Height,
	unsigned char* rgbImage,
	unsigned short alpha = 255,
	bool inverted = false
);
void hav_p016_HDR10_bgra64_HDR10(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int p016Width,
	unsigned int p016Height,
	unsigned short* rgbImage,
	bool inverted = false,
	unsigned short alpha = 1023
);
