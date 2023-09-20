package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/net"
)

type Attention struct {
	Module
	wq *tensor.Tensor
	wk *tensor.Tensor
	wv *tensor.Tensor
	wo *tensor.Tensor
}

var _ layer.Layer = &Attention{}

func NewAttention(wq, wk, wv, wo *tensor.Tensor) *Attention {
	return &Attention{
		wq: wq,
		wk: wk,
		wv: wv,
		wo: wo,
	}
}

func init() {
	net.RegisterLoadFunc("llama2.attention", func(name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer {
		var layer Attention
		layer.wq = params["wq"]
		layer.wk = params["wk"]
		layer.wv = params["wv"]
		layer.wo = params["wo"]
		return &layer
	})
}

func (l *Attention) Class() string {
	return "llama2.attention"
}

func (l *Attention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"wq": l.wq,
		"wk": l.wk,
		"wv": l.wv,
		"wo": l.wo,
	}
}

func (l *Attention) Freeze() {
	l.wq.SetRequiresGrad(false)
	l.wk.SetRequiresGrad(false)
	l.wv.SetRequiresGrad(false)
	l.wo.SetRequiresGrad(false)
}

func (l *Attention) Unfreeze() {
	l.wq.SetRequiresGrad(true)
	l.wk.SetRequiresGrad(true)
	l.wv.SetRequiresGrad(true)
	l.wo.SetRequiresGrad(true)
}

func (l *Attention) ToScalarType(t consts.ScalarType) *Attention {
	var layer Attention
	layer.wq = l.wq.ToScalarType(t)
	layer.wk = l.wk.ToScalarType(t)
	layer.wv = l.wv.ToScalarType(t)
	layer.wo = l.wo.ToScalarType(t)
	return &layer
}
