package model

import "llama2/internal/model/layer"

type block struct {
	attn     *layer.Attention
	attnNorm *layer.RMSNorm
	ffn      *feedforward
	ffnNorm  *layer.RMSNorm
}
