package model

import (
	"fmt"
	ilayer "llama2/internal/model/layer"

	"github.com/lwch/gotorch/consts"
	gmodel "github.com/lwch/gotorch/model"
	"github.com/lwch/tnn/nn/net"
)

type Model struct {
	embedding *ilayer.Embedding
	blocks    []*block
	norm      *ilayer.RMSNorm
	output    *ilayer.Linear
}

func LoadFromTorch(m *gmodel.Model, params *Params) *Model {
	var md Model
	md.embedding = ilayer.NewEmbedding(m.Get("tok_embeddings.weight"))
	for i := 0; i < params.Layers; i++ {
		wk := m.Get(fmt.Sprintf("layers.%d.attention.wk.weight", i))
		wq := m.Get(fmt.Sprintf("layers.%d.attention.wq.weight", i))
		wv := m.Get(fmt.Sprintf("layers.%d.attention.wv.weight", i))
		wo := m.Get(fmt.Sprintf("layers.%d.attention.wo.weight", i))
		norm1 := m.Get(fmt.Sprintf("layers.%d.attention.norm", i))
		w1 := m.Get(fmt.Sprintf("layers.%d.feed_forward.w1.weight", i))
		w2 := m.Get(fmt.Sprintf("layers.%d.feed_forward.w2.weight", i))
		w3 := m.Get(fmt.Sprintf("layers.%d.feed_forward.w3.weight", i))
		norm2 := m.Get(fmt.Sprintf("layers.%d.ffn_norm.norm", i))

		var block block
		block.attn = ilayer.NewAttention(wq, wk, wv, wo)
		block.attnNorm = ilayer.NewRMSNorm(norm1)
		block.ffn = newFeedforward(w1, w2, w3)
		block.ffnNorm = ilayer.NewRMSNorm(norm2)

		md.blocks = append(md.blocks, &block)
	}
	md.norm = ilayer.NewRMSNorm(m.Get("norm.weight"))
	md.output = ilayer.NewLinear(m.Get("output.weight"))
	return &md
}

func (m *Model) ToScalarType(t consts.ScalarType) *Model {
	var model Model
	model.embedding = m.embedding.ToScalarType(t)
	return &model
}

func (m *Model) Save(dir string) error {
	var net net.Net
	net.Add(m.embedding)
	for _, block := range m.blocks {
		net.Add(block.attn)
		net.Add(block.attnNorm)
		net.Add(block.ffn)
		net.Add(block.ffnNorm)
	}
	net.Add(m.norm, m.output)
	return net.Save(dir)
}
