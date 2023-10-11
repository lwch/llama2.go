package model

import (
	"llama2/internal/tensor"

	"github.com/lwch/logging"
	"github.com/lwch/sentencepiece"
)

type Model struct {
	embeddingWeight tensor.Param
	attentionWQ     []tensor.Param
	attentionWK     []tensor.Param
	attentionWV     []tensor.Param
	attentionWO     []tensor.Param
	attentionNorm   []tensor.Param
	ffnW1           []tensor.Param
	ffnW2           []tensor.Param
	ffnW3           []tensor.Param
	ffnNorm         []tensor.Param
	norm            tensor.Param
	output          tensor.Param
	embeddingDim    int
	layers          int
	heads           int
	eps             float32
	tk              *sentencepiece.Model
}

func (m *Model) ShowInfo() {
	logging.Info("model info:")
	logging.Info("  + embedding dim: %d", m.embeddingDim)
	logging.Info("  + layers: %d", m.layers)
	logging.Info("  + heads: %d", m.heads)
}

func (m *Model) GetTokenizer() *sentencepiece.Model {
	return m.tk
}
