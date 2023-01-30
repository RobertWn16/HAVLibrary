#pragma once
#include <algorithm>

void hav_nv12_bgra32_SDR(unsigned char* SDRLuma,
	unsigned char* SDRChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float wr,
	float wb,
	unsigned char* rgbImage,
	bool exAlpha,
	unsigned int alpha
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
	bool useFP16,
	unsigned int nits,
	bool inverted = false,
	unsigned short alpha = 1023
);


void hav_p016_HDR10_bgra64_HDR10_PQ_ACES(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float max_content_luminance,
	float display_luminance,
	float wr,
	float wb,
	unsigned short* HDRRGBA,
	bool exAlpha,
	unsigned int alpha
);

void hav_p016_HDR10_bgra64_HDR10_PQ_Reinhard(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float max_content_luminance,
	float display_luminance,
	float wr,
	float wb,
	unsigned short* HDRRGBA,
	bool exAlpha,
	unsigned int alpha
);

void hav_p016_HDR10_bgra64_HDR10_Linear(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float display_luminance,
	float wr,
	float wb,
	unsigned short* HDRRGBA,
	bool exAlpha,
	unsigned int alpha
);
