package model

import "llama2/internal/tensor"

type Model struct {
	embeddingWeight tensor.Tensor
	attentionWQ     []tensor.Tensor
	attentionWK     []tensor.Tensor
	attentionWV     []tensor.Tensor
	attentionWO     []tensor.Tensor
	attentionNorm   []tensor.Tensor
	ffnW1           []tensor.Tensor
	ffnW2           []tensor.Tensor
	ffnW3           []tensor.Tensor
	ffnNorm         []tensor.Tensor
	norm            tensor.Tensor
	output          tensor.Tensor
}
