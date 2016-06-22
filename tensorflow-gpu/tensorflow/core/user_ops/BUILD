load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
)

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)

tf_custom_op_library(
    name = "my_pad.so",
    srcs = ["my_pad.cc"],
    gpu_srcs = ["my_pad.cu.cc"]
)

# py_library(
#     name = "my_pad_op",
#     srcs = ["my_pad_op.py"],
#     data = [":my_pad.so"],
#     srcs_version = "PY2AND3",
# )

py_test(
    name = "my_pad_op_test",
    size = "small",
    srcs = ["my_pad_op_test.py"],
    srcs_version = "PY2AND3",
    data = [":my_pad.so"],
    deps = [
#        ":my_pad_op",
         "//tensorflow:tensorflow_py",
    ],
)
