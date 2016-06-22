#define EIGEN_USE_THREADS

#include <memory>
#include <string>
#include <utility>
#include <typeinfo>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

// #include "tensorflow/core/kernels/pad_op.h"


REGISTER_OP("MPad")
.Attr("T: type")
.Input("input: T")
.Input("paddings: int32")
.Output("output: T")
;

// template <typename Device, typename T>
// void MyPadOpExec(const Device& d, typename tensorflow::TTypes<T, 1>::Tensor output,
//                  typename tensorflow::TTypes<T, 1>::ConstTensor input,
//                  Eigen::array<std::pair<tensorflow::int32, tensorflow::int32>, 1> paddings) {

//     std::cout << "is cpu" << std::endl;
//     output.device(d) = input.pad(paddings);
// }

template <typename Device, typename T>
void MyPadOpExec(const Device& d, typename tensorflow::TTypes<T, 1>::Tensor output,
                 typename tensorflow::TTypes<T, 1>::ConstTensor input,
                 Eigen::array<std::pair<tensorflow::int32, tensorflow::int32>, 1> paddings);


namespace tensorflow {

 
    // namespace functor {
    //     // Functor used by PadOp to do the computations.
    //     template <typename Device, typename T>
    //     void exec(const Device& d, typename TTypes<T, 1>::Tensor output,
    //                   typename TTypes<T, 1>::ConstTensor input,
    //                   Eigen::array<std::pair<int32, int32>, 1> paddings) {

    //         std::cout << "is cpu" << std::endl;
    //         output.device(d) = input.pad(paddings);
    //     }
    // } // namespace functor

    typedef Eigen::ThreadPoolDevice CPUDevice;
    typedef Eigen::GpuDevice GPUDevice;

    template <typename Device, typename T>
    class MyPadOp : public OpKernel {
    public:
        explicit MyPadOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            const Tensor& in0 = context->input(0);
            std::cout << in0.DebugString() << std::endl;

            const Tensor& in1 = context->input(1);
            const int dims = in0.dims();
            static const int dimSize = 1;
            OP_REQUIRES(context, 1 == dims,
                        errors::Unimplemented("inputs rank not in [", dimSize, ",]: ", dims));
            OP_REQUIRES(
                        context,
                        TensorShapeUtils::IsMatrix(in1.shape()) && in1.dim_size(1) == 2,
                        errors::InvalidArgument("paddings must be a matrix with 2 columns: ",
                                                in1.shape().DebugString()));
            const int fixed_dims = 1;
            OP_REQUIRES(
                        context, fixed_dims == in1.dim_size(0),
                        errors::InvalidArgument(
                                                "The first dimension of paddings must be the rank of inputs",
                                                in1.shape().DebugString(), " ", in0.shape().DebugString()));

            // Compute the shape of the output tensor, and allocate it.
            TensorShape output_shape;
            TTypes<int32>::ConstMatrix paddings = in1.matrix<int32>();
            for (int d = 0; d < fixed_dims; ++d) {
                const int32 before_d = paddings(d, 0);  // Pad before existing elements.
                const int32 after_d = paddings(d, 1);   // Pad after existing elements.
                OP_REQUIRES(context, before_d >= 0 && after_d >= 0,
                            errors::InvalidArgument("Paddings must be non-negative: ",
                                                    before_d, " ", after_d));
                const int64 size_d =
                    (allow_legacy_scalars() && d == in0.dims()) ? 1 : in0.dim_size(d);
                output_shape.AddDim(before_d + size_d + after_d);
            }
            Tensor* output = nullptr;

            std::cout << "output shape:" <<  output_shape.dims()  << std::endl;
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

            // change flat to tensor.  That is, once !allow_legacy_scalars().
            Operate(context, in0.flat<T>(), paddings, output);
        }

    private:
        void Operate(OpKernelContext* context,
                     typename TTypes<T, 1>::ConstTensor input,
                     TTypes<int32>::ConstMatrix paddings, Tensor* output) {
            CHECK_EQ(1, paddings.dimension(0));
            CHECK_EQ(2, paddings.dimension(1));
            Eigen::array<std::pair<int32, int32>, 1> paddings_array;
            for (int i = 0; i < 1; ++i) {
                paddings_array[i] = std::make_pair(paddings(i, 0), paddings(i, 1));
            }

            const Device& d = context->eigen_device<Device>();
            // std::cout << d << std::endl;
            typename TTypes<T, 1>::Tensor output_tensor = output->tensor<T, 1>();

            // functor::exec<Device, T>(d, output_tensor, input, paddings_array);

            MyPadOpExec<Device, T>(d, output_tensor, input, paddings_array);
            // output_tensor.device(d) = input.pad(paddings_array);
        }
    };


#define REGISTER_KERNEL(type)                               \
        REGISTER_KERNEL_BUILDER(Name("MPad")                \
                                .Device(DEVICE_CPU)         \
                                .TypeConstraint<type>("T")  \
                                .HostMemory("paddings"),    \
                                MyPadOp<CPUDevice, type>)

    TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL



#if GOOGLE_CUDA

#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("MPad")                 \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          MyPadOp<GPUDevice, T>)


    //TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_GPU_KERNEL);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);

#endif  // GOOGLE_CUDA

}
// end namespace tensorflow
