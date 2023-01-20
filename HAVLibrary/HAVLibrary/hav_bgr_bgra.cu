#include "hav_bgr_bgra.cuh"
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda/std/cmath>
#include <device_launch_parameters.h>

#define  Pr  .299
#define  Pg  .587
#define  Pb  .114

constexpr double LHDR = 13.25f;
constexpr double LSDR = 5.6969f;
__device__ double Mat_YUV422_RGB[3][3] =
{
	{1.0f, 0.0f, 1.402f},
	{1.0f, -0.344f, -0.714f},
	{1.0f, 1.722f, 0.0f}
};

__device__ double Mat_BT2020[3][3] =
{
	{1.0f, 0.0f, 1.4747f},
	{1.0f, -0.571f, -0.1645f},
	{1.0f, 1.8814f, 0.0f}
};

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
	return x < lower ? lower : (x > upper ? upper : x);
}
template<typename Out>
__global__ void hav_p016_bgra64_kernel(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int width,
	unsigned int heigth,
	Out* destImage,
	bool inverted,
	bool exAlpha,
	unsigned int alpha)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	int resolution = width * heigth;
	unsigned int pitch = width;
	unsigned int pitchFactor = 3;
	for (int i = index; i < resolution; i += stride)
	{
		unsigned int curRow = i / width;
		unsigned int curColumn = i - curRow * width;
		unsigned int u_idx = curRow / 2 * width + curColumn;
		u_idx -= u_idx & 1;
		unsigned int v_idx = u_idx + 1;


		int Y = (cuLuma[i] >> 8) - 16;
		int U = (cuChroma[u_idx] >> 8) - 128;
		int V = (cuChroma[v_idx] >> 8) - 128;

		int R = Mat_BT2020[0][0] * Y + Mat_BT2020[0][2] * V;
		int G = Mat_BT2020[1][0] * Y + Mat_BT2020[1][1] * U + Mat_BT2020[1][2] * V;
		int B = Mat_BT2020[2][0] * Y + Mat_BT2020[2][1] * U;


		if (exAlpha)
			pitchFactor = 4;

		if (inverted) {
			destImage[pitchFactor * i + 0] = Clamp((int)R, 0, 255);
			destImage[pitchFactor * i + 1] = Clamp((int)G, 0, 255);
			destImage[pitchFactor * i + 2] = Clamp((int)B, 0, 255);
		}
		else {
			destImage[pitchFactor * i + 0] = Clamp((int)B, 0, 255);
			destImage[pitchFactor * i + 1] = Clamp((int)G, 0, 255);
			destImage[pitchFactor * i + 2] = Clamp((int)R, 0, 255);
		}

		if (exAlpha)
			destImage[pitchFactor * i + 3] = 255;


		/*int Y = (cuLuma[i] >> 6) - 64;
		int U = (cuChroma[u_idx] >> 6) - 512;
		int V = (cuChroma[v_idx] >> 6) - 512;

		double R = Mat_BT2020[0][0] * Y + Mat_BT2020[0][2] * V;
		double G = Mat_BT2020[1][0] * Y + Mat_BT2020[1][1] * U + Mat_BT2020[1][2] * V;
		double B = Mat_BT2020[2][0] * Y + Mat_BT2020[2][1] * U;



		double Yp = 0;
		double Yc = 0;
		//Using gamma correction, not fully optimized
		double norm = 1023;
		double R_norm = 0;
		double G_norm = 0;
		double B_norm = 0;
		if (norm) {
			R_norm = R / norm;
			G_norm = G / norm;
			B_norm = B / norm;
		}

		double R_prime = pow(R_norm, 1.0f / 2.4f);
		double G_prime = pow(G_norm, 1.0f / 2.4f);
		double B_prime = pow(B_norm, 1.0f / 2.4f);
		double Y_prime = 0.2627f * R_prime + 0.6780f * G_prime + 0.0593f * B_prime;
		Yp = log10(1 + (LHDR - 1) * (double)Y_prime) / log10(LHDR);


		if (Yp >= 0.0f && Yp <= 0.74f)
			Yc = 1.0770f * Yp;

		if (Yp > 0.74f && Yp <= 0.9909f)
			Yc = -1.1510f * Yp * Yp + 2.7811f * Yp - 0.6302;

		if (Yp > 0.9912f && Yp <= 1)
			Yc = 0.5f * Yp + 0.5f;

		double YSDR = (pow(LSDR, Yc) - 1) / (LSDR - 1);
		double fYSDR = YSDR / (1.1 * Y_prime);

		double U_sdr = fYSDR * (double)(R_prime - Y_prime) / 1.8814f;
		double V_sdr = fYSDR * (double)(B_prime - Y_prime) / 1.4746f;
		double val = 0.1f * V_sdr;
		double Y_sdr = YSDR - fmax(val, (double)0.0f);

		double rTmp = Mat_BT2020[0][0] * Y_sdr + Mat_BT2020[0][2] * V_sdr;
		double gTmp = Mat_BT2020[1][0] * Y_sdr + Mat_BT2020[1][1] * U_sdr + Mat_BT2020[1][2] * V_sdr;
		double bTmp = Mat_BT2020[2][0] * Y_sdr + Mat_BT2020[2][1] * U_sdr;

		rTmp *= 255;
		gTmp *= 255;
		bTmp *= 255;*/

	}
	return;
}

template <typename Out>
__global__ void hav_nv12_bgra64_kernel(unsigned char* cuLuma,
	unsigned char* cuChroma,
	unsigned int width,
	unsigned int heigth, 
	Out* destImage,
	bool inverted,
	bool exAlpha,
	unsigned int alpha
)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	int resolution = width * heigth;
	unsigned int pitch = width;
	unsigned int pitchFactor = 3;

	if (exAlpha)
		pitchFactor = 4;

	for (int i = index; i < resolution; i += stride)
	{
		unsigned int curRow = i / width;
		unsigned int curColumn = i - curRow * width;
		unsigned int u_idx = curRow / 2 * width + curColumn;
		u_idx -= u_idx & 1;
		unsigned int v_idx = u_idx + 1;
		int maxValue = (1 << (sizeof(Out) * 8)) - 1;

		int Y = (int)cuLuma[i]- 16;
		int U = (int)cuChroma[u_idx] - 128;
		int V = (int)cuChroma[v_idx] - 128;

		int R = Mat_YUV422_RGB[0][0] * Y + Mat_YUV422_RGB[0][1] * U + Mat_YUV422_RGB[0][2] * V;
		int G = Mat_YUV422_RGB[1][0] * Y + Mat_YUV422_RGB[1][1] * U + Mat_YUV422_RGB[1][2] * V;
		int B = Mat_YUV422_RGB[2][0] * Y + Mat_YUV422_RGB[2][1] * U + Mat_YUV422_RGB[2][2] * V;
	
		if (inverted) {
			destImage[pitchFactor * i + 0] = Clamp(R, 0, maxValue); // Limited RGB Saturation
			destImage[pitchFactor * i + 1] = Clamp(G, 0, maxValue);
			destImage[pitchFactor * i + 2] = Clamp(B, 0, maxValue);
		}
		else {
			destImage[pitchFactor * i + 2] = Clamp(R, 0, maxValue); // Limited RGB Saturation
			destImage[pitchFactor * i + 1] = Clamp(G, 0, maxValue);
			destImage[pitchFactor * i + 0] = Clamp(B, 0, maxValue);
		}

		if (exAlpha)
			destImage[pitchFactor * i + 3] = alpha;
	}

	return;
}

void hav_nv12_bgra32_SDR(unsigned char* SDRLuma,
	unsigned char* SDRChroma,
	unsigned int width,
	unsigned int heigth,
	unsigned char* rgbImage,
	unsigned int alpha,
	bool inverted
)
{
	hav_nv12_bgra64_kernel << <320, 180 >> > (SDRLuma, SDRChroma, width, heigth, rgbImage, inverted, true, alpha);
}

void hav_p016_bgra64(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int width,
	unsigned int heigth,
	unsigned char* destImage,
	bool inverted,
	bool exAlpha, 
	unsigned int alpha)
{
		
}

void hav_p016_HDR10_bgr32_SDR(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int p016Width, 
	unsigned int p016Height,
	unsigned char* rgbImage,
	bool inverted)
{
	
}

void hav_p016_HDR10_bgr64_HDR10(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int p016Width,
	unsigned int p016Height,
	unsigned short* rgbImage,
	bool inverted)
{

}

void hav_p016_HDR10_bgra32_SDR(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int p016Width,
	unsigned int p016Height, 
	unsigned char* rgbImage,
	unsigned short alpha,
	bool inverted
)
{
	hav_p016_bgra64_kernel << <320, 180 >> > (HDRLuma, HDRChroma, p016Width, p016Height, rgbImage, inverted, true, alpha);
}

void hav_p016_HDR10_bgra64_HDR10(unsigned short* HDRLuma,
	unsigned short* HDRChroma,
	unsigned int p016Width,
	unsigned int p016Height,
	unsigned short* rgbImage,
	bool inverted,
	unsigned short alpha
)
{
}
