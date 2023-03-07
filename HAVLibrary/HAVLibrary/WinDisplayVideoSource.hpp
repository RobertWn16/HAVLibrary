#pragma once
#include "IVIdeoSource.hpp"
#include <dxgi1_6.h>
#include <d3d11.h>

struct WinDisplayVideoSource : winrt::implements<WinDisplayVideoSource, IVideoSource>
{
private:
	VIDEO_SOURCE_DESC video_source_desc;
	ID3D11Device* pwdsDevice;
	ID3D11DeviceContext* pwdsDeviceCtx;
	IDXGIOutput6* pwdsOutput6;
	winrt::com_ptr<IDXGIOutputDuplication> pwdsOutputDupl;
	winrt::com_ptr<ID3D11Texture2D> pwdsTex;
public:
	winrt::hresult GetDesc(VIDEO_SOURCE_DESC& desc);
	winrt::hresult Parse(void* desc);
	winrt::hresult Parse(ID3D11Texture2D **Out) noexcept final;
	winrt::hresult ConfigureVideoSource(VIDEO_SOURCE_DESC vsrc_desc, ID3D11Device* pDevice, IDXGIOutput6* pOutputDupl);
};