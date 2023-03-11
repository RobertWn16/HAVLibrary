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

void hav_p016_HDR10_bgra64_HDR10_PQ_ACES(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float RGB_XYZ_Matrix[3][3],
	float XYZ_RGB_Matrix[3][3],
	float max_content_luminance,
	float display_luminance,
	unsigned short* HDRRGBA,
	bool exAlpha,
	unsigned int alpha
);

void hav_p016_HDR10_bgra64_HDR10_PQ_Reinhard(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float RGB_XYZ_Matrix[3][3],
	float XYZ_RGB_Matrix[3][3],
	float max_content_luminance,
	float display_luminance,
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

void hav_p016_HDR10_bgra32_SDR_Linear(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float wr,
	float wb,
	unsigned char* SDRImage,
	bool exAlpha,
	unsigned int alpha
);

void hav_bgr24_bgra32_SDR(unsigned char* bgr,
	unsigned int width,
	unsigned int height,
	unsigned int alpha,
	unsigned char* bgra
);