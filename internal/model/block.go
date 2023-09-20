package model

import (
	"llama2/internal/model/layer"

	"github.com/lwch/gotorch/consts"
)

type block struct {
	attn     *layer.Attention
	attnNorm *layer.RMSNorm
	ffn      *feedforward
	ffnNorm  *layer.RMSNorm
}

func (b *block) toScalarType(t consts.ScalarType) *block {
	var block block
	block.attn = b.attn.ToScalarType(t)
	block.attnNorm = b.attnNorm.ToScalarType(t)
	block.ffn = b.ffn.toScalarType(t)
	block.ffnNorm = b.ffnNorm.ToScalarType(t)
	return &block
}
