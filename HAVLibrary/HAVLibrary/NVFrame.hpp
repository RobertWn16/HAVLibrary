#pragma once
#include "IFrame.hpp"
#include "HAVUtilsPrivate.hpp"
#include <d3d11.h>
#include <cuda_d3d11_interop.h>

struct NVFrame : winrt::implements<NVFrame, IFrame>
{
private:
	cudaGraphicsResource* cuResource;
	float content_Colorimetry_XYZ[3][3];
	float content_Colorimetry_XYZ_Inverse[3][3];
	float display_Colorimetry_XYZ[3][3];
	float display_Colrimetry_XYZ_Inverse[3][3];
	float Colorimetry_Convertor[3][3];

	winrt::hresult ComputeColorimetryMatrix(HVColorimetry in_colorimetry, float XYZ[3][3], float XYZ_Inverse[3][3]);
	void ComputeMatrixInverse(float mat[3][3], float mat_inv[3][3]);
public:
	FRAME_OUTPUT_DESC cuDesc;
	CUdeviceptr cuFrame;
	unsigned int cuFrameSize;
	cudaArray_t resourceArray = nullptr;

	~NVFrame();

	winrt::hresult GetDesc(FRAME_OUTPUT_DESC& desc);
	winrt::hresult ConvertFormat(HVFormat fmt, IFrame *out);
	winrt::hresult RegisterD3D11Resource(ID3D11Resource* resource);
	winrt::hresult CommitResource();
	winrt::hresult ConfigureFrame(FRAME_OUTPUT_DESC desc);
};