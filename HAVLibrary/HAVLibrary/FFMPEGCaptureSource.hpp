#include "IDemuxer.hpp"

extern "C"
{
	#include <libavformat/avformat.h>
	#include <libavcodec/avcodec.h>
	#include <libavutil/avutil.h>
	#include <libavcodec/bsf.h>
}

struct FVContext
{
public:
	AVFormatContext* av_fmt_context = nullptr;
	AVInputFormat* av_inp_format = nullptr;
	AVBSFContext* av_bsf_context = nullptr;
	const AVBitStreamFilter* av_bsf_filter = nullptr;
	AVStream* vstream = nullptr;
};

struct FFMPEGCaptureSource : winrt::implements<FFMPEGCaptureSource, IVCaptureSource>
{
private:
	FVContext source_ctx;
	VIDEO_SOURCE_DESC source_desc;

public:
	winrt::hresult GetDesc(VIDEO_SOURCE_DESC &desc) final;

public:
	void InitSource(FVContext ctx, VIDEO_SOURCE_DESC desc);
};

