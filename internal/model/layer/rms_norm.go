package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/net"
)

type RMSNorm struct {
	Module
	w   *tensor.Tensor
	eps *tensor.Tensor
}

var _ layer.Layer = &RMSNorm{}

func NewRMSNorm(w, eps *tensor.Tensor) *RMSNorm {
	eps.SetRequiresGrad(false)
	return &RMSNorm{
		w:   w,
		eps: eps,
	}
}

func (l *RMSNorm) norm(x *tensor.Tensor) *tensor.Tensor {
	return x.Mul(x.Pow(2).Mean(-1, true).Add(l.eps).RSqrt())
}

func (l *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	y := l.norm(x.ToScalarType(consts.KFloat)).ToScalarType(x.ScalarType())
	return y.Mul(l.w)
}

func init() {
	net.RegisterLoadFunc("llama2.rmsnorm", func(name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer {
		var layer RMSNorm
		layer.w = params["w"]
		layer.eps = params["eps"]
		layer.eps.SetRequiresGrad(false)
		return &layer
	})
}

func (l *RMSNorm) Class() string {
	return "llama2.rmsnorm"
}

func (l *RMSNorm) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w":   l.w,
		"eps": l.eps,
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
