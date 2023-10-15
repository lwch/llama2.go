package model

import (
	"llama2/internal/param"

	"github.com/lwch/logging"
	"github.com/lwch/sentencepiece"
)

type Model struct {
	embeddingWeight param.Param
	attentionWQ     []param.Param
	attentionWK     []param.Param
	attentionWV     []param.Param
	attentionWO     []param.Param
	attentionNorm   []param.Param
	ffnW1           []param.Param
	ffnW2           []param.Param
	ffnW3           []param.Param
	ffnNorm         []param.Param
	norm            param.Param
	output          param.Param
	embeddingDim    int64
	layers          int
	heads           int64
	kvHeads         int64
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
