#include <torch/extension.h>
#include "render.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_tets", &RenderFTetsCUDA);
    m.def("render_tets_backward", &RenderFTetsBackwardCUDA);
}