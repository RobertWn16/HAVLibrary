#include "IFrame.hpp"
#include "HAVUtilsPrivate.hpp"
#include <d3d11.h>
#include <cuda_d3d11_interop.h>

struct NVFrame : winrt::implements<NVFrame, IFrame>
{
private:
	cudaGraphicsResource* cuResource;
	
	void NV12_BGRX8(NVFrame* out, bool inverted, bool exAlpha = false, int value = 255);
	void P016_BGRX8(NVFrame* out, bool inverted, bool exAlpha = false, int value = 255);
public:
	winrt::hresult GetDesc(FRAME_OUTPUT_DESC& desc);
	winrt::hresult ConvertFormat(HVFormat fmt, IFrame *out);
	winrt::hresult RegisterD3D11Resource(ID3D11Resource* resource);
	winrt::hresult CommitResource();

	winrt::hresult ConfigureFrame(FRAME_OUTPUT_DESC desc);
public:
	FRAME_OUTPUT_DESC cuDesc;
	CUdeviceptr cuFrame;
	unsigned int cuFrameSize;
	cudaArray_t resourceArray = nullptr;
};