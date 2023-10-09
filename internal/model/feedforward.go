package model

import (
	ilayer "llama2/internal/model/layer"
	"llama2/internal/model/parallel"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/net"
)

type feedforward struct {
	ilayer.Module
	w1 *tensor.Tensor
	w2 *tensor.Tensor
	w3 *tensor.Tensor
}

var _ layer.Layer = &feedforward{}

func newFeedforward(w1, w2, w3 *tensor.Tensor) *feedforward {
	return &feedforward{
		w1: w1,
		w2: w2,
		w3: w3,
	}
}

func (l *feedforward) forward(x *tensor.Tensor) *tensor.Tensor {
	left := parallel.MatMul(x, l.w1).Silu()
	right := parallel.MatMul(x, l.w3)
	x = left.Mul(right)
	return parallel.MatMul(x, l.w2)
}

func init() {
	net.RegisterLoadFunc("llama2.ffn", func(name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer {
		var layer feedforward
		layer.w1 = params["w1"]
		layer.w2 = params["w2"]
		layer.w3 = params["w3"]
		return &layer
	})
}

func (l *feedforward) Class() string {
	return "llama2.ffn"
}

func (l *feedforward) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w1": l.w1,
		"w2": l.w2,
		"w3": l.w3,
	}
}

func (l *feedforward) Freeze() {
	l.w1.SetRequiresGrad(false)
	l.w2.SetRequiresGrad(false)
	l.w3.SetRequiresGrad(false)
}

func (l *feedforward) Unfreeze() {
	l.w1.SetRequiresGrad(true)
	l.w2.SetRequiresGrad(true)
	l.w3.SetRequiresGrad(true)
}

func (l *feedforward) toScalarType(t consts.ScalarType) *feedforward {
	var layer feedforward
	layer.w1 = l.w1.ToScalarType(t)
	layer.w2 = l.w2.ToScalarType(t)
	layer.w3 = l.w3.ToScalarType(t)
	return &layer
}
