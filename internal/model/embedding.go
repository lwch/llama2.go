package model

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/net"
)

type embeddingLayer struct {
	module
	w *tensor.Tensor
}

var _ layer.Layer = &embeddingLayer{}

func newEmbeddingLayer(w *tensor.Tensor) *embeddingLayer {
	return &embeddingLayer{
		w: w,
	}
}

func init() {
	net.RegisterLoadFunc("llama2.embedding", func(name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer {
		var layer embeddingLayer
		layer.w = params["w"]
		return &layer
	})
}

func (l *embeddingLayer) Class() string {
	return "llama2.embedding"
}

func (l *embeddingLayer) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": l.w,
	}
}

func (l *embeddingLayer) Freeze() {
	l.w.SetRequiresGrad(false)
}

func (l *embeddingLayer) Unfreeze() {
	l.w.SetRequiresGrad(true)
}

func (l *embeddingLayer) ToScalarType(t consts.ScalarType) *embeddingLayer {
	var layer embeddingLayer
	layer.w = l.w.ToScalarType(t)
	return &layer
}
