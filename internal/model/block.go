package model

import (
	"llama2/internal/model/layer"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type block struct {
	attn     *layer.Attention
	attnNorm *layer.RMSNorm
	ffn      *feedforward
	ffnNorm  *layer.RMSNorm
}

func (b *block) forward(x *tensor.Tensor) *tensor.Tensor {
	x = b.attn.Forward(x)
	x = b.attnNorm.Forward(x)
	x = b.ffn.forward(x)
	return b.ffnNorm.Forward(x)
}

func (b *block) toScalarType(t consts.ScalarType) *block {
	var block block
	block.attn = b.attn.ToScalarType(t)
	block.attnNorm = b.attnNorm.ToScalarType(t)
	block.ffn = b.ffn.toScalarType(t)
	block.ffnNorm = b.ffnNorm.ToScalarType(t)
	return &block
}
