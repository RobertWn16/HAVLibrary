#include "NVDEC.hpp"
int main(int argc, char** argv)
{
	winrt::com_ptr<IDecoder> nv = winrt::make_self<NVDEC>();
	nv.get()->TestFunction();

	return 0;
}