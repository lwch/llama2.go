package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/net"
)

type Embedding struct {
	Module
	w *tensor.Tensor
}

var _ layer.Layer = &Embedding{}

func NewEmbedding(w *tensor.Tensor) *Embedding {
	return &Embedding{
		w: w,
	}
}

func init() {
	net.RegisterLoadFunc("llama2.embedding", func(name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer {
		var layer Embedding
		layer.w = params["w"]
		return &layer
	})
}

func (l *Embedding) Class() string {
	return "llama2.embedding"
}

func (l *Embedding) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": l.w,
	}
}

func (l *Embedding) Freeze() {
	l.w.SetRequiresGrad(false)
}

func (l *Embedding) Unfreeze() {
	l.w.SetRequiresGrad(true)
}

func (l *Embedding) ToScalarType(t consts.ScalarType) *Embedding {
	var layer Embedding
	layer.w = l.w.ToScalarType(t)
	return &layer
}
