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
	HV_CODEC_MJPEG,
	HV_CODEC_UNSUPPORTED
};

enum HVChroma
{
	HV_CHROMA_FORMAT_420 = 0,
	HV_CHROMA_FORMAT_422,
	HV_CHROMA_FORMAT_444,
	HV_CHROMA_FORMAT_UNKNOWN
};

enum HVFormat
{
	HV_FORMAT_UNKNOWN = 0,
	HV_FORMAT_RGBA8,
	HV_FORMAT_RGB8,
	HV_FORMAT_BGR24,
	HV_FORMAT_BGRA32,
	HV_FORMAT_BGRA64_HDR10,
	HV_FORMAT_NV12,
	HV_FORMAT_P016,
};

enum HVColorSpace
{
	HV_COLORSPACE_UNKNOWN = 0,
	HV_COLORSPACE_BT601,
	HV_COLORSPACE_BT709,
	HV_COLORSPACE_ADOBE_RGB,
	HV_COLORSPACE_BT2020,
	HV_COLORSPACE_DISPLAY_P3,
	HV_COLORSPACE_CUSTOM
};

enum HVTransfer
{
	HV_TRANSFER_UNKNOWN = 0,
	HV_TRANSFER_LINEAR,
	HV_TRANSFER_PQ,
	HV_TRANSFER_HLG,
};

enum HVToneMapper
{
	HV_TONE_MAPPER_UNKNOWN = 0,
	HV_TONE_MAPPER_REINHARD_EXT,
	HV_TONE_MAPPER_ACES
};

enum HAVError
{
	E_HV_NOT_LINKED = ERROR_GRAPHICS_ONLY_CONSOLE_SESSION_SUPPORTED + 1,
	E_HV_ALREADY_LINKED = E_HV_NOT_LINKED + 1,
	E_HV_CODEC_NOT_SUPPORTED = E_HV_ALREADY_LINKED + 1,
};

struct HVColorimetry
{
	float xr;
	float yr;
	float xg;
	float yg;
	float xb;
	float yb;
	float xw;
	float yw;
};

const HVColorimetry BT601_Colorimetry_625lines = { .xr = 0.640f, .yr = 0.630, .xg = 0.290, .yg = 0.600f, .xb = 0.15f, .yb = 0.060f, .xw = 0.3127f, .yw = 0.3290 };
const HVColorimetry BT709_Colorimetry = { .xr = 0.64f, .yr = 0.33f, .xg = 0.3f, .yg = 0.6f, .xb = 0.15f, .yb = 0.06f, .xw = 0.3127f, .yw = 0.3290f};
const HVColorimetry BT2020_Colorimetry = { .xr = 0.708f, .yr = 0.292f, .xg = 0.170f, .yg = 0.797f, .xb = 0.131f, .yb = 0.046f, .xw = 0.3127f, .yw = 0.3290f };
const HVColorimetry DisplayP3_Colorimetry = { .xr = 0.680f, .yr = 0.320f, .xg = 0.265f, .yg = 0.690f, .xb = 0.150f, .yb = 0.060f, .xw = 0.3127f, .yw = 0.3290f };