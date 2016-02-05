#ifndef CAFFE_PADDING_LAYER_HPP_
#define CAFFE_PADDING_LAYER_HPP_


namespace caffe {
/**
 * @brief Pads (if pad >= 0) or crops (if pad < 0) parts of the input.
 */
template <typename Dtype>
class PaddingLayer : public Layer<Dtype> {
 public:
  explicit PaddingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int pad_beg_, pad_end_;
  bool pad_pos_;
  int num_, channels_;
  int height_in_, width_in_;
  int height_out_, width_out_;
};

} //namespace caffe

#endif // CAFFE_PADDING_LAYER_HPP_
