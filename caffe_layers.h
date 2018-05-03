/*
layers register
*/
#include "caffe/common.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/permute_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/prior_box_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/detection_output_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/bias_layer.hpp"

namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
//	REGISTER_LAYER_CLASS(Input);
	extern INSTANTIATE_CLASS(InnerProductLayer);
//	REGISTER_LAYER_CLASS(InnerProduct);
	extern INSTANTIATE_CLASS(DropoutLayer);
//	REGISTER_LAYER_CLASS(Dropout);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);
	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);
	extern INSTANTIATE_CLASS(LRNLayer);
	REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTER_LAYER_CLASS(Softmax);

	extern INSTANTIATE_CLASS(NormalizeLayer);
//	REGISTER_LAYER_CLASS(Normalize);
	extern INSTANTIATE_CLASS(PermuteLayer);
//	REGISTER_LAYER_CLASS(Permute);
	extern INSTANTIATE_CLASS(FlattenLayer);
//	REGISTER_LAYER_CLASS(Flatten);
	extern INSTANTIATE_CLASS(PriorBoxLayer);
//	REGISTER_LAYER_CLASS(PriorBox);
	extern INSTANTIATE_CLASS(ReshapeLayer);
//	REGISTER_LAYER_CLASS(Reshape);
	extern INSTANTIATE_CLASS(ConcatLayer);
//	REGISTER_LAYER_CLASS(Concat);
	extern INSTANTIATE_CLASS(DetectionOutputLayer);
//	REGISTER_LAYER_CLASS(DetectionOutput);
	extern INSTANTIATE_CLASS(BatchNormLayer);
//	REGISTER_LAYER_CLASS(BatchNorm);
	extern INSTANTIATE_CLASS(DeconvolutionLayer);
//	REGISTER_LAYER_CLASS(Deconvolution);
	extern INSTANTIATE_CLASS(DepthwiseConvolutionLayer);
//	REGISTER_LAYER_CLASS(DepthwiseConvolution);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
}