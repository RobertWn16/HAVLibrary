#include "ColorSpaceConversion.cuh"
#include "LUTS.cuh"
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda/std/cmath>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

constexpr float SDR_NITS = 80.0f;

//ACES curve constants
__device__ float aces_a = 2.51f;
__device__ float aces_b = 0.03f;
__device__ float aces_c = 2.43f;
__device__ float aces_d = 0.59f;
__device__ float aces_e = 0.14f;

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
	return x < lower ? lower : (x > upper ? upper : x);
}

__device__ float __device_RGB_XYZ_Mat[3][3];
__device__ float __device_XYZ_RGB_Mat[3][3];

__device__ inline unsigned short float2half(float f)
{
	union fasi { float f; unsigned int i; };
	fasi v;
	v.f = f;

	unsigned short sign = (v.i >> 31) & 0x1;
	short exp = ((v.i >> 23) & 0xff) - 127;
	unsigned short mant = (v.i >> 13) & 0x3ff;

	if (exp < -14)
	{
		exp = 0;
		mant = (0x400 & mant) >> (-exp - 14);
	}
	else if (exp < 16)
	{
		exp += 15;
	}
	else
	{
		// just make it inf, ignore NaN
		exp = 31;
		mant = 0;
	}

	return (sign << 15) | (exp << 10) | mant;
}

__global__ void p016_HDR10_bgra64_HDR10_PQ_ACES_kernel(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float max_content_luminance,
	float display_luminance,
	float XYZ_RGB_Matrix[3][3],
	float RGB_XYZ_Matrix[3][3],
	float wr,
	float wb,
	float wg,
	float wgb,
	float wgr,
	float wr_coef,
	float wb_coef,
	float white_black_coeff,
	unsigned short* destImage,
	bool exAlpha,
	unsigned int alpha)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	int resolution = width * heigth;
	unsigned int pitchFactor = 3;


	float display_lum_coeff = display_luminance / SDR_NITS;
	unsigned int shift_depth = 6;
	unsigned int Y_offset = 64;
	unsigned int UV_offset = 512;
	int maxValue = 1023.0f;

	int Y = 0.0f;
	int U = 0.0f;
	int V = 0.0f;

	float X_plane = 0.0f;
	float Y_plane = 0.0f;
	float Z_plane = 0.0f;

	float R_FP32 = 0.0f;
	float G_FP32 = 0.0f;
	float B_FP32 = 0.0f;
	half A_FP32 = 0.0f;
	if (exAlpha) {
		pitchFactor = 4;
		A_FP32 = (float)alpha;
	}

	unsigned int curRow = 0;
	unsigned int curColumn = 0;
	unsigned int u_idx = 0;
	unsigned int v_idx = 0;

	for (int i = index; i < resolution; i += stride)
	{
		curRow = i / width;
		curColumn = i - curRow * width;
		u_idx = curRow / 2 * width + curColumn;
		u_idx -= u_idx & 1;
		v_idx = u_idx + 1;

		Y = (cuLuma[i] >> shift_depth) - Y_offset;
		U = (cuChroma[u_idx] >> shift_depth) - UV_offset;
		V = (cuChroma[v_idx] >> shift_depth) - UV_offset;

		R_FP32 = (Y + wr_coef * V) * white_black_coeff;
		B_FP32 = (Y + wb_coef * U) * white_black_coeff;
		G_FP32 = ((Y - wr * R_FP32 - wb * B_FP32) / wg) * white_black_coeff;

		R_FP32 = R_FP32 < 0 ? 0 : (R_FP32 > (maxValue - 1) ? (maxValue - 1) : R_FP32);
		G_FP32 = G_FP32 < 0 ? 0 : (G_FP32 > (maxValue - 1) ? (maxValue - 1) : G_FP32);
		B_FP32 = B_FP32 < 0 ? 0 : (B_FP32 > (maxValue - 1) ? (maxValue - 1) : B_FP32);

		R_FP32 = EOTF_LUT[(int)R_FP32];
		G_FP32 = EOTF_LUT[(int)G_FP32];
		B_FP32 = EOTF_LUT[(int)B_FP32];

		R_FP32 = R_FP32 < 0 ? 0 : (R_FP32 > (maxValue - 1) ? (maxValue - 1) : R_FP32);
		G_FP32 = G_FP32 < 0 ? 0 : (G_FP32 > (maxValue - 1) ? (maxValue - 1) : G_FP32);
		B_FP32 = B_FP32 < 0 ? 0 : (B_FP32 > (maxValue - 1) ? (maxValue - 1) : B_FP32);

		X_plane = __device_RGB_XYZ_Mat[0][0] * R_FP32 + __device_RGB_XYZ_Mat[0][1] * G_FP32 + __device_RGB_XYZ_Mat[0][2] * B_FP32;
		Y_plane = __device_RGB_XYZ_Mat[1][0] * R_FP32 + __device_RGB_XYZ_Mat[1][1] * G_FP32 + __device_RGB_XYZ_Mat[1][2] * B_FP32;
		Z_plane = __device_RGB_XYZ_Mat[2][0] * R_FP32 + __device_RGB_XYZ_Mat[2][1] * G_FP32 + __device_RGB_XYZ_Mat[2][2] * B_FP32;

		R_FP32 = __device_XYZ_RGB_Mat[0][0] * X_plane + __device_XYZ_RGB_Mat[0][1] * Y_plane + __device_XYZ_RGB_Mat[0][2] * Z_plane;
		G_FP32 = __device_XYZ_RGB_Mat[1][0] * X_plane + __device_XYZ_RGB_Mat[1][1] * Y_plane + __device_XYZ_RGB_Mat[1][2] * Z_plane;
		B_FP32 = __device_XYZ_RGB_Mat[2][0] * X_plane + __device_XYZ_RGB_Mat[2][1] * Y_plane + __device_XYZ_RGB_Mat[2][2] * Z_plane;

		R_FP32 = (R_FP32 * max_content_luminance) / SDR_NITS;
		G_FP32 = (G_FP32 * max_content_luminance) / SDR_NITS;
		B_FP32 = (B_FP32 * max_content_luminance) / SDR_NITS;

		//3ms
		R_FP32 = R_FP32 * (aces_a * R_FP32 + aces_b) / (R_FP32 * (aces_c * R_FP32 + aces_d) + aces_e) * display_lum_coeff;
		G_FP32 = G_FP32 * (aces_a * G_FP32 + aces_b) / (G_FP32 * (aces_c * G_FP32 + aces_d) + aces_e) * display_lum_coeff;
		B_FP32 = B_FP32 * (aces_a * B_FP32 + aces_b) / (B_FP32 * (aces_c * B_FP32 + aces_d) + aces_e) * display_lum_coeff;

		destImage[pitchFactor * i + 0] = __float2half(R_FP32).operator __half_raw().x;
		destImage[pitchFactor * i + 1] = __float2half(G_FP32).operator __half_raw().x;
		destImage[pitchFactor * i + 2] = __float2half(B_FP32).operator __half_raw().x;

		if (exAlpha)
			destImage[pitchFactor * i + 3] = A_FP32.operator __half_raw().x;
	}
	return;
}

__device__ float ApplySRGBCurve_Fast(float x)
{
	return x < 0.0031308 ? 12.92 * x : 1.13005 * sqrt(x - 0.00228) - 0.13448 * x + 0.005719;
}
__global__ void p016_HDR10_bgra64_HDR10_PQ_Reinhard_kernel(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float max_content_luminance,
	float display_luminance,
	float wr,
	float wb,
	float wg,
	float wgb,
	float wgr,
	float wr_coef,
	float wb_coef,
	float white_black_coeff,
	unsigned short* destImage,
	bool exAlpha,
	unsigned int alpha
)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	int resolution = width * heigth;
	unsigned int pitchFactor = 3;


	float display_lum_coeff = display_luminance / SDR_NITS;
	unsigned int shift_depth = 6;
	unsigned int Y_offset = 64;
	unsigned int UV_offset = 512;
	int maxValue = 1023.0f;

	int Y = 0.0f;
	int U = 0.0f;
	int V = 0.0f;

	float X_plane = 0.0f;
	float Y_plane = 0.0f;
	float Z_plane = 0.0f;

	float R_FP32 = 0.0f;
	float G_FP32 = 0.0f;
	float B_FP32 = 0.0f;
	half A_FP32 = 0.0f;
	if (exAlpha) {
		pitchFactor = 4;
		A_FP32 = (float)alpha;
	}

	unsigned int curRow = 0;
	unsigned int curColumn = 0;
	unsigned int u_idx = 0;
	unsigned int v_idx = 0;

	for (int i = index; i < resolution; i += stride)
	{
		curRow = i / width;
		curColumn = i - curRow * width;
		u_idx = curRow / 2 * width + curColumn;
		u_idx -= u_idx & 1;
		v_idx = u_idx + 1;

		Y = (cuLuma[i] >> shift_depth) - Y_offset;
		U = (cuChroma[u_idx] >> shift_depth) - UV_offset;
		V = (cuChroma[v_idx] >> shift_depth) - UV_offset;

		R_FP32 = (Y + wr_coef * V) * white_black_coeff;
		B_FP32 = (Y + wb_coef * U) * white_black_coeff;
		G_FP32 = ((Y - wr * R_FP32 - wb * B_FP32) / wg) * white_black_coeff;

		R_FP32 = R_FP32 < 0 ? 0 : (R_FP32 > (maxValue - 1) ? (maxValue - 1) : R_FP32);
		G_FP32 = G_FP32 < 0 ? 0 : (G_FP32 > (maxValue - 1) ? (maxValue - 1) : G_FP32);
		B_FP32 = B_FP32 < 0 ? 0 : (B_FP32 > (maxValue - 1) ? (maxValue - 1) : B_FP32);

		R_FP32 = EOTF_LUT[(int)R_FP32];
		G_FP32 = EOTF_LUT[(int)G_FP32];
		B_FP32 = EOTF_LUT[(int)B_FP32];

		X_plane = __device_RGB_XYZ_Mat[0][0] * R_FP32 + __device_RGB_XYZ_Mat[0][1] * G_FP32 + __device_RGB_XYZ_Mat[0][2] * B_FP32;
		Y_plane = __device_RGB_XYZ_Mat[1][0] * R_FP32 + __device_RGB_XYZ_Mat[1][1] * G_FP32 + __device_RGB_XYZ_Mat[1][2] * B_FP32;
		Z_plane = __device_RGB_XYZ_Mat[2][0] * R_FP32 + __device_RGB_XYZ_Mat[2][1] * G_FP32 + __device_RGB_XYZ_Mat[2][2] * B_FP32;

		R_FP32 = __device_XYZ_RGB_Mat[0][0] * X_plane + __device_XYZ_RGB_Mat[0][1] * Y_plane + __device_XYZ_RGB_Mat[0][2] * Z_plane;
		G_FP32 = __device_XYZ_RGB_Mat[1][0] * X_plane + __device_XYZ_RGB_Mat[1][1] * Y_plane + __device_XYZ_RGB_Mat[1][2] * Z_plane;
		B_FP32 = __device_XYZ_RGB_Mat[2][0] * X_plane + __device_XYZ_RGB_Mat[2][1] * Y_plane + __device_XYZ_RGB_Mat[2][2] * Z_plane;

		R_FP32 = (R_FP32 * max_content_luminance) / SDR_NITS;
		G_FP32 = (G_FP32 * max_content_luminance) / SDR_NITS;
		B_FP32 = (B_FP32 * max_content_luminance) / SDR_NITS;

		R_FP32 = (R_FP32 / (R_FP32 + 1)) * display_lum_coeff;
		G_FP32 = (G_FP32 / (G_FP32 + 1)) * display_lum_coeff;
		B_FP32 = (B_FP32 / (B_FP32 + 1)) * display_lum_coeff;

		destImage[pitchFactor * i + 0] = __float2half(R_FP32).operator __half_raw().x;
		destImage[pitchFactor * i + 1] = __float2half(G_FP32).operator __half_raw().x;
		destImage[pitchFactor * i + 2] = __float2half(B_FP32).operator __half_raw().x;

		if (exAlpha)
			destImage[pitchFactor * i + 3] = A_FP32.operator __half_raw().x;
	}
	return;
}

__global__ void p016_HDR10_bgra64_HDR10_Linear_kernel(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float wr,
	float wb,
	float wg,
	float wgb,
	float wgr,
	float wr_coef,
	float wb_coef,
	float display_luminance,
	float white_black_coeff,
	unsigned short* destImage,
	bool exAlpha,
	unsigned int alpha
)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	int resolution = width * heigth;
	unsigned int pitchFactor = 3;

	unsigned int shift_depth = 6;
	unsigned int Y_offset = 64;
	unsigned int UV_offset = 512;
	float display_lum_coeff = display_luminance / SDR_NITS;
	int maxValue = 1023.0f;

	int Y = 0.0f;
	int U = 0.0f;
	int V = 0.0f;

	float R_FP32 = 0.0f;
	float G_FP32 = 0.0f;
	float B_FP32 = 0.0f;
	half A_FP32 = 0.0f;
	if (exAlpha) {
		pitchFactor = 4;
		A_FP32 = (float)alpha;
	}

	unsigned int curRow = 0;
	unsigned int curColumn = 0;
	unsigned int u_idx = 0;
	unsigned int v_idx = 0;

	for (int i = index; i < resolution; i += stride)
	{
		curRow = i / width;
		curColumn = i - curRow * width;
		u_idx = curRow / 2 * width + curColumn;
		u_idx -= u_idx & 1;
		v_idx = u_idx + 1;

		Y = (cuLuma[i] >> shift_depth) - Y_offset;
		U = (cuChroma[u_idx] >> shift_depth) - UV_offset;
		V = (cuChroma[v_idx] >> shift_depth) - UV_offset;

		R_FP32 = (Y + wr_coef * V) * white_black_coeff;
		B_FP32 = (Y + wb_coef * U) * white_black_coeff;
		G_FP32 = ((Y - wr * R_FP32 - wb * B_FP32) / wg) * white_black_coeff;

		R_FP32 = (Clamp(R_FP32, 0.0f, (float)maxValue) / maxValue) * display_lum_coeff;
		G_FP32 = (Clamp(G_FP32, 0.0f, (float)maxValue) / maxValue) * display_lum_coeff;
		B_FP32 = (Clamp(B_FP32, 0.0f, (float)maxValue) / maxValue) * display_lum_coeff;

		destImage[pitchFactor * i + 0] = float2half(R_FP32);
		destImage[pitchFactor * i + 1] = float2half(G_FP32);
		destImage[pitchFactor * i + 2] = float2half(B_FP32);

		if (exAlpha)
			destImage[pitchFactor * i + 3] = A_FP32.operator __half_raw().x;
	}

	return;
}

__global__ void p016_HDR10_bgra32_SDR_Linear_kernel(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float wr,
	float wb,
	float wg,
	float wgb,
	float wgr,
	float wr_coef,
	float wb_coef,
	float white_black_coeff,
	unsigned char* destImage,
	bool exAlpha,
	unsigned int alpha
)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	int resolution = width * heigth;
	unsigned int pitch = width;
	unsigned int pitchFactor = 3;

	unsigned int shift_depth = 8;
	unsigned int Y_offset = 16;
	unsigned int UV_offset = 128;
	if (exAlpha)
		pitchFactor = 4;
	
	int maxValue = 255;

	int Y = 0;
	int U = 0;
	int V = 0;

	float R_FP32 = 0.0f;
	float G_FP32 = 0.0f;
	float B_FP32 = 0.0f;

	for (int i = index; i < resolution; i += stride)
	{
		unsigned int curRow = i / width;
		unsigned int curColumn = i - curRow * width;
		unsigned int u_idx = curRow / 2 * width + curColumn;
		u_idx -= u_idx & 1;
		unsigned int v_idx = u_idx + 1;


		Y = (cuLuma[i] >> shift_depth) - Y_offset;
		U = (cuChroma[u_idx] >> shift_depth) - UV_offset;
		V = (cuChroma[v_idx] >> shift_depth) - UV_offset;

		R_FP32 = (Y + wr_coef * V) * white_black_coeff;
		B_FP32 = (Y + wb_coef * U) * white_black_coeff;
		G_FP32 = ((Y - wr * R_FP32 - wb * B_FP32) / wg) * white_black_coeff;

		if (inverted) {
			destImage[pitchFactor * i + 0] = Clamp((int)R_FP32, 0, maxValue); // Limited RGB Saturation
			destImage[pitchFactor * i + 1] = Clamp((int)G_FP32, 0, maxValue);
			destImage[pitchFactor * i + 2] = Clamp((int)B_FP32, 0, maxValue);
		}
		else {
			destImage[pitchFactor * i + 2] = Clamp((int)R_FP32, 0, maxValue); // Limited RGB Saturation
			destImage[pitchFactor * i + 1] = Clamp((int)G_FP32, 0, maxValue);
			destImage[pitchFactor * i + 0] = Clamp((int)B_FP32, 0, maxValue);
		}

		if (exAlpha)
			destImage[pitchFactor * i + 3] = alpha;
	}

	return;
}

__global__ void nv12_SDR_bgra32_SDR_kernel(unsigned char* cuLuma,
	unsigned char* cuChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float wr,
	float wb,
	float wg,
	float wgb,
	float wgr,
	float wr_coef,
	float wb_coef,
	float white_black_coeff,
	unsigned char* destImage,
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
	int maxValue = 255.0f;

	int Y = 0;
	int U = 0;
	int V = 0;

	float R_FP32 = 0.0f;
	float G_FP32 = 0.0f;
	float B_FP32 = 0.0f;

	for (int i = index; i < resolution; i += stride)
	{
		unsigned int curRow = i / width;
		unsigned int curColumn = i - curRow * width;
		unsigned int u_idx = curRow / 2 * width + curColumn;
		u_idx -= u_idx & 1;
		unsigned int v_idx = u_idx + 1;


		Y = (int)cuLuma[i] - 16;
		U = (int)cuChroma[u_idx] - 128;
		V = (int)cuChroma[v_idx] - 128;

		R_FP32 = (Y + wr_coef * V) * white_black_coeff;
		B_FP32 = (Y + wb_coef * U) * white_black_coeff;
		G_FP32 = ((Y - wr * R_FP32 - wb * B_FP32) / wg) * white_black_coeff;
	
		if (inverted) {
			destImage[pitchFactor * i + 0] = Clamp((int)R_FP32, 0, maxValue); // Limited RGB Saturation
			destImage[pitchFactor * i + 1] = Clamp((int)G_FP32, 0, maxValue);
			destImage[pitchFactor * i + 2] = Clamp((int)B_FP32, 0, maxValue);
		}
		else {
			destImage[pitchFactor * i + 2] = Clamp((int)R_FP32, 0, maxValue); // Limited RGB Saturation
			destImage[pitchFactor * i + 1] = Clamp((int)G_FP32, 0, maxValue);
			destImage[pitchFactor * i + 0] = Clamp((int)B_FP32, 0, maxValue);
		}

		if (exAlpha)
			destImage[pitchFactor * i + 3] = alpha;
	}

	return;
}

__global__ void bgr24_bgra32_kernel(unsigned char* srcImage,
	unsigned int width,
	unsigned int heigth,
	unsigned int alpha,
	unsigned char* destImage
)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int resolution = width * heigth;
	for (unsigned int i = index; i < width * heigth; i += stride)
	{
		destImage[4 * i + 0] = srcImage[i * 3 + 0];
		destImage[4 * i + 1] = srcImage[i * 3 + 1];
		destImage[4 * i + 2] = srcImage[i * 3 + 2];
		destImage[4 * i + 3] = Clamp((int)alpha, 0, 255);
	}
}


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
)
{
	float wg = 1 - wr - wb;
	float wgb = -wb * (1.0f - wb) / 0.5f / (1 - wb - wr);
	float wgr = -wr * (1 - wr) / 0.5f / (1 - wb - wr);
	float white_black_coeff = 1.16f;

	float wr_coef = (1.0f - wr) / 0.5f;
	float wb_coef = (1.0f - wb) / 0.5f;

	nv12_SDR_bgra32_SDR_kernel << <320, 180 >> > (SDRLuma,
		SDRChroma,
		width,
		heigth,
		inverted,
		wr,
		wb,
		wg,
		wgb,
		wgr,
		wr_coef,
		wb_coef,
		white_black_coeff,
		rgbImage,
		true,
		alpha);
}

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
	unsigned int alpha)
{
	float wr = RGB_XYZ_Matrix[1][0];
	float wb = RGB_XYZ_Matrix[1][2];
	float wg = 1.0f - wr - wb;
	float wgb = -wb * (1.0f - wb) / 0.5f / wg;
	float wgr = -wr * (1 - wr) / 0.5f / wg;
	float white_black_coeff = 1.16f;

	float wr_coef = (1.0f - RGB_XYZ_Matrix[1][0]) / 0.5f;
	float wb_coef = (1.0f - RGB_XYZ_Matrix[1][2]) / 0.5f;

	//no check
	cudaMemcpyToSymbol(__device_RGB_XYZ_Mat, RGB_XYZ_Matrix, 9 * sizeof(float));
	cudaMemcpyToSymbol(__device_XYZ_RGB_Mat, XYZ_RGB_Matrix, 9 * sizeof(float));

	p016_HDR10_bgra64_HDR10_PQ_ACES_kernel << <1240, 360 >> > (HDRLuma,
		HDRChroma,
		width,
		heigth,
		inverted,
		max_content_luminance,
		display_luminance,
		XYZ_RGB_Matrix,
		RGB_XYZ_Matrix,
		wr,
		wb,
		wg,
		wgb,
		wgr,
		wr_coef,
		wb_coef,
		1.16f,
		HDRRGBA,
		exAlpha,
		alpha);
}


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
	unsigned int alpha)
{
	float wr = RGB_XYZ_Matrix[1][0];
	float wb = RGB_XYZ_Matrix[1][2];
	float wg = 1.0f - wr - wb;
	float wgb = -wb * (1.0f - wb) / 0.5f / wg;
	float wgr = -wr * (1 - wr) / 0.5f / wg;
	float white_black_coeff = 1.16f;

	float wr_coef = (1.0f - RGB_XYZ_Matrix[1][0]) / 0.5f;
	float wb_coef = (1.0f - RGB_XYZ_Matrix[1][2]) / 0.5f;

	//no check
	cudaMemcpyToSymbol(__device_RGB_XYZ_Mat, RGB_XYZ_Matrix, 9 * sizeof(float));
	cudaMemcpyToSymbol(__device_XYZ_RGB_Mat, XYZ_RGB_Matrix, 9 * sizeof(float));

	p016_HDR10_bgra64_HDR10_PQ_Reinhard_kernel << <1240, 360 >> > (HDRLuma,
		HDRChroma,
		width,
		heigth,
		inverted,
		max_content_luminance,
		display_luminance,
		wr,
		wb,
		wg,
		wgb,
		wgr,
		wr_coef,
		wb_coef,
		1.16f,
		HDRRGBA,
		exAlpha,
		alpha);
}

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
	unsigned int alpha)
{
	float wg = 1.0f - wr - wb;
	float wgb = -wb * (1.0f - wb) / 0.5f / (1 - wb - wr);
	float wgr = -wr * (1 - wr) / 0.5f / (1 - wb - wr);
	float white_black_coeff = 1.16f;

	float wr_coef = (1.0f - wr) / 0.5f;
	float wb_coef = (1.0f - wb) / 0.5f;

	p016_HDR10_bgra64_HDR10_Linear_kernel << <1240, 360 >> > (HDRLuma,
		HDRChroma,
		width,
		heigth,
		inverted,
		wr,
		wb,
		wg,
		wgb,
		wgr,
		wr_coef,
		wb_coef,
		display_luminance,
		1.16f,
		HDRRGBA,
		exAlpha,
		alpha);
}

void hav_p016_HDR10_bgra32_SDR_Linear(unsigned short* cuLuma,
	unsigned short* cuChroma,
	unsigned int width,
	unsigned int heigth,
	bool inverted,
	float wr,
	float wb,
	unsigned char* destImage,
	bool exAlpha,
	unsigned int alpha
)
{
	float wg = 1.0f - wr - wb;
	float wgb = -wb * (1.0f - wb) / 0.5f / (1 - wb - wr);
	float wgr = -wr * (1 - wr) / 0.5f / (1 - wb - wr);
	float white_black_coeff = 1.16f;

	float wr_coef = (1.0f - wr) / 0.5f;
	float wb_coef = (1.0f - wb) / 0.5f;

	p016_HDR10_bgra32_SDR_Linear_kernel << <1240, 360 >> > (cuLuma,
		cuChroma,
		width,
		heigth,
		inverted,
		wr,
		wb,
		wg,
		wgb,
		wgr,
		wr_coef,
		wb_coef,
		white_black_coeff,
		destImage,
		exAlpha,
		alpha
	);
}

void hav_bgr24_bgra32_SDR(unsigned char* bgr, unsigned int width, unsigned int height, unsigned int alpha, unsigned char* bgra)
{
	bgr24_bgra32_kernel << <320, 180 >> > (bgr, width, height, alpha, bgra);

}

