#pragma once
#include "IPacket.hpp"
#include "IDecoder.hpp"
#include "HAVUtilsPrivate.hpp"
#include "NVFrame.hpp"
#include <nvjpeg.h>

#pragma comment (lib, "nvjpeg")
static winrt::hresult NVJPEGHr(nvjpegStatus_t nvjpeg_stat)
{
	if (nvjpeg_stat != NVJPEG_STATUS_SUCCESS)
		return E_INVALIDARG;
	return S_OK;
}
struct NVJpegDecoder : winrt::implements<NVJpegDecoder, IDecoder>
{
private:
	nvjpegHandle_t nvHandle;
	nvjpegJpegState_t nvjpegState;
	cudaStream_t stream;
	nvjpegImage_t nvjpegImage;
	unsigned int width;
	unsigned int height;

public:
	IVideoSource* vSource;
	~NVJpegDecoder();

	winrt::hresult IsSupported(VIDEO_SOURCE_DESC video_src_desc) final;
	winrt::hresult Decode(IPacket* in, IFrame* out) final;
	winrt::hresult Decode(unsigned char* buf, unsigned int length, unsigned int timestamp, IFrame* out);
	winrt::hresult CreateNVJpeg(VIDEO_SOURCE_DESC video_src_desc) noexcept;

public:
	winrt::hresult ConfigureDecoder(VIDEO_SOURCE_DESC vsrc_desc, IDev *dev) final;
};