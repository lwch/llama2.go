package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/net"
)

type RMSNorm struct {
	Module
	w *tensor.Tensor
}

var _ layer.Layer = &RMSNorm{}

func NewRMSNorm(w *tensor.Tensor) *RMSNorm {
	return &RMSNorm{
		w: w,
	}
}

func init() {
	net.RegisterLoadFunc("llama2.rmsnorm", func(name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer {
		var layer Attention
		layer.wq = params["w"]
		return &layer
	})
}

func (l *RMSNorm) Class() string {
	return "llama2.rmsnorm"
}

func (l *RMSNorm) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": l.w,
	}
}

func (l *RMSNorm) Freeze() {
	l.w.SetRequiresGrad(false)
}

func (l *RMSNorm) Unfreeze() {
	l.w.SetRequiresGrad(true)
}

func (l *RMSNorm) ToScalarType(t consts.ScalarType) *RMSNorm {
	var layer RMSNorm
	layer.w = l.w.ToScalarType(t)
	return &layer
}
