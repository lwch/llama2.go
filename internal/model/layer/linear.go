package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/net"
)

type Linear struct {
	Module
	w *tensor.Tensor
}

var _ layer.Layer = &Linear{}

func NewLinear(w *tensor.Tensor) *Linear {
	return &Linear{
		w: w,
	}
}

func init() {
	net.RegisterLoadFunc("llama2.linear", func(name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer {
		var layer Linear
		layer.w = params["w"]
		return &layer
	})
}

func (l *Linear) Class() string {
	return "llama2.linear"
}

func (l *Linear) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": l.w,
	}
}

func (l *Linear) Freeze() {
	l.w.SetRequiresGrad(false)
}

func (l *Linear) Unfreeze() {
	l.w.SetRequiresGrad(true)
}

func (l *Linear) ToScalarType(t consts.ScalarType) *Linear {
	var layer Linear
	layer.w = l.w.ToScalarType(t)
	return &layer
}
