package model

import (
	"fmt"
	ilayer "llama2/internal/model/layer"

	"github.com/lwch/gotorch/consts"
	gmodel "github.com/lwch/gotorch/model"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/net"
)

type Model struct {
	embedding *ilayer.Embedding
	blocks    []*block
	norm      *ilayer.RMSNorm
	output    *ilayer.Linear
}

func LoadFromTorch(m *gmodel.Model, params *Params) *Model {
	getParam := func(m *gmodel.Model, name string) *tensor.Tensor {
		t := m.Get(name)
		if t == nil {
			panic(fmt.Sprintf("cannot find %s", name))
		}
		return t
	}

	eps := tensor.FromFloat32(nil, []float32{params.Eps}, tensor.WithShapes(1))

	var md Model
	md.embedding = ilayer.NewEmbedding(getParam(m, "tok_embeddings.weight"))
	for i := 0; i < params.Layers; i++ {
		wk := getParam(m, fmt.Sprintf("layers.%d.attention.wk.weight", i))
		wq := getParam(m, fmt.Sprintf("layers.%d.attention.wq.weight", i))
		wv := getParam(m, fmt.Sprintf("layers.%d.attention.wv.weight", i))
		wo := getParam(m, fmt.Sprintf("layers.%d.attention.wo.weight", i))
		norm1 := getParam(m, fmt.Sprintf("layers.%d.attention_norm.weight", i))
		w1 := getParam(m, fmt.Sprintf("layers.%d.feed_forward.w1.weight", i))
		w2 := getParam(m, fmt.Sprintf("layers.%d.feed_forward.w2.weight", i))
		w3 := getParam(m, fmt.Sprintf("layers.%d.feed_forward.w3.weight", i))
		norm2 := getParam(m, fmt.Sprintf("layers.%d.ffn_norm.weight", i))

		var block block
		block.attn = ilayer.NewAttention(wq, wk, wv, wo)
		block.attnNorm = ilayer.NewRMSNorm(norm1, eps)
		block.ffn = newFeedforward(w1, w2, w3)
		block.ffnNorm = ilayer.NewRMSNorm(norm2, eps)

		md.blocks = append(md.blocks, &block)
	}
	md.norm = ilayer.NewRMSNorm(getParam(m, "norm.weight"), eps)
	md.output = ilayer.NewLinear(getParam(m, "output.weight"))
	params.Vocabs = int(getParam(m, "output.weight").Shapes()[0])
	return &md
}

func LoadFromTNN(dir string, params *Params) *Model {
	var net net.Net
	err := net.Load(dir)
	runtime.Assert(err)
	layers := net.Layers()

	var md Model
	var idx int
	md.embedding = layers[idx].(*ilayer.Embedding)
	idx++
	for i := 0; i < params.Layers; i++ {
		var block block
		block.attn = layers[idx].(*ilayer.Attention)
		idx++
		block.attnNorm = layers[idx].(*ilayer.RMSNorm)
		idx++
		block.ffn = layers[idx].(*feedforward)
		idx++
		block.ffnNorm = layers[idx].(*ilayer.RMSNorm)
		idx++
		md.blocks = append(md.blocks, &block)
	}
	md.norm = layers[idx].(*ilayer.RMSNorm)
	idx++
	md.output = layers[idx].(*ilayer.Linear)
	return &md
}

func (m *Model) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = m.embedding.Forward(x)
	for _, block := range m.blocks {
		x = block.forward(x)
	}
	x = m.norm.Forward(x)
	return m.output.Forward(x)
}

func (m *Model) ToScalarType(t consts.ScalarType) *Model {
	var model Model
	model.embedding = m.embedding.ToScalarType(t)
	for _, block := range m.blocks {
		model.blocks = append(model.blocks, block.toScalarType(t))
	}
	model.norm = m.norm.ToScalarType(t)
	model.output = m.output.ToScalarType(t)
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
