#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

// #include "tensorflow/core/kernels/pad_op.h"


template <typename Device, typename T>
void MyPadOpExec(const Device& d, typename tensorflow::TTypes<T, 1>::Tensor output,
                 typename tensorflow::TTypes<T, 1>::ConstTensor input,
                 Eigen::array<std::pair<tensorflow::int32, tensorflow::int32>, 1> paddings) {

    std::cout << "is gpu" << std::endl;
    To32Bit(output).device(d) = To32Bit(input).pad(paddings);

    // std::cout << "is cpu" << std::endl;
    // output.device(d) = input.pad(paddings);
}


// namespace tensorflow {

//     namespace functor {

//         // Functor used by PadOp to do the computations.
//         template <typename Device, typename T>
//         void exec(const Device& d, typename TTypes<T, 1>::Tensor output,
//                       typename TTypes<T, 1>::ConstTensor input,
//                       Eigen::array<std::pair<int32, int32>, 1> paddings) {
//             std::cout << "is gpu" << std::endl;
//             To32Bit(output).device(d) = To32Bit(input).pad(paddings);
//         }
//     }
// }

    
// end namespace tensorflow


#endif  // GOOGLE_CUDA

