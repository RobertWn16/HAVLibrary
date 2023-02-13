#include "NVENC.hpp"

#pragma comment (lib, "nvencodeapi")
static winrt::hresult NVENCStatHr(NVENCSTATUS nvenc_stat)
{
    if (nvenc_stat != NV_ENC_SUCCESS)
        return E_INVALIDARG;
    return S_OK;
}
winrt::hresult NVENC::IsSupported(VIDEO_SOURCE_DESC desc)
{
    return winrt::hresult();
}

winrt::hresult NVENC::Encode(IFrame* out)
{
    return winrt::hresult();
}

winrt::hresult NVENC::ConfigureEncoder(CUcontext deviceContext)
{
    try
    {
        NVENCSTATUS stat = NV_ENC_SUCCESS;
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS nvencOpenPar = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
        uint32_t numOfGuids;
        GUID guids[5];
        GUID presetGuids[20];
        NV_ENC_TUNING_INFO nvencTunInfo;
        NV_ENC_PRESET_CONFIG nvencPresetConf = { NV_ENC_PRESET_CONFIG_VER, {NV_ENC_CONFIG_VER} };
        nvencPresetConf.presetCfg.version = NV_ENC_CONFIG_VER;
        NV_ENC_BUFFER_FORMAT format[30];
        NV_ENC_CONFIG nvencConf;
        uint32_t inputFormatCount;
        NV_ENC_INITIALIZE_PARAMS nvencParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        uint32_t numOfPreset;

        memset(&nvencParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
        winrt::check_pointer(&deviceContext);
        winrt::check_hresult(NVENCStatHr(NvEncodeAPICreateInstance(&functionList)));
        nvencOpenPar.apiVersion = NVENCAPI_VERSION;
        nvencOpenPar.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
        nvencOpenPar.device = deviceContext;
        winrt::check_hresult(NVENCStatHr(functionList.nvEncOpenEncodeSessionEx(&nvencOpenPar, &nvencEncoder)));

        functionList.nvEncGetEncodeGUIDCount(nvencEncoder, &numOfGuids);
        functionList.nvEncGetEncodeGUIDs(nvencEncoder, guids, 5, &numOfGuids);

        stat = functionList.nvEncGetEncodePresetCount(nvencEncoder, guids[10], &numOfPreset);
        stat = functionList.nvEncGetEncodePresetGUIDs(nvencEncoder, guids[0], presetGuids, 20, &numOfPreset);
        stat = functionList.nvEncGetEncodePresetConfigEx(nvencEncoder, guids[0], NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO_LOSSLESS, &nvencPresetConf);
        stat = functionList.nvEncGetInputFormatCount(nvencEncoder, guids[0], &inputFormatCount);
        stat = functionList.nvEncGetInputFormats(nvencEncoder, guids[0], format, 30, &inputFormatCount);

        nvencParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
        nvencParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
        nvencParams.encodeWidth = 1920;
        nvencParams.encodeHeight = 1080;
        nvencParams.maxEncodeHeight = 2160;
        nvencParams.maxEncodeWidth = 3840;
        nvencParams.darWidth = 1920;
        nvencParams.darHeight = 1080;
        nvencParams.enableEncodeAsync = false;
        nvencParams.encodeConfig = &nvencPresetConf.presetCfg;
        stat = functionList.nvEncInitializeEncoder(nvencEncoder, &nvencParams);


    }catch (winrt::hresult_error& err) {
        return err.code();
    }
    return S_OK;
}
