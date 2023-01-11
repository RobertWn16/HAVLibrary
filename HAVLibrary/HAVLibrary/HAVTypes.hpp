#pragma once
enum HVCodec
{
	HV_CODEC_MPEG1 = 0,
	HV_CODEC_MPEG2,
	HV_CODEC_VC1,
	HV_CODEC_VP8,
	HV_CODEC_VP9,
	HV_CODEC_H264,
	HV_CODEC_H265_420,
	HV_CODEC_H265_444,
	HV_CODEC_AV1,
	HV_CODEC_UNSUPPORTED
};

enum HVChroma
{
	HV_CHROMA_FORMAT_420 = 0,
	HV_CHROMA_FORMAT_422,
	HV_CHROMA_FORMAT_444
};

enum HAVError
{
	E_HV_NOT_LINKED = ERROR_GRAPHICS_ONLY_CONSOLE_SESSION_SUPPORTED + 1,
	E_HV_ALREADY_LINKED = E_HV_NOT_LINKED + 1,
	E_HV_CODEC_NOT_SUPPORTED = E_HV_ALREADY_LINKED + 1,
};