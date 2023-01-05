#include "IDecoder.hpp"

struct NVDEC : winrt::implements<NVDEC, IDecoder>
{
public:
	winrt::hresult TestFunction() final;
};