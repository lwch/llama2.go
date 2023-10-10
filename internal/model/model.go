package model

import (
	"fmt"
	ilayer "llama2/internal/model/layer"
	"math"

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
	freqs     *tensor.Tensor
}

func LoadFromTorch(m *gmodel.Model, params *Params) *Model {
	getParam := func(m *gmodel.Model, name string) *tensor.Tensor {
		t := m.Get(name)
		if t == nil {
			panic(fmt.Sprintf("cannot find %s", name))
		}
		return t
	}

	eps := tensor.FromBFloat16(nil, []float32{params.Eps}, tensor.WithShapes(1))

	var md Model
	md.embedding = ilayer.NewEmbedding(getParam(m, "tok_embeddings.weight"))
	for i := 0; i < params.Layers; i++ {
		wk := getParam(m, fmt.Sprintf("layers.%d.attention.wk.weight", i)).Transpose(0, 1).Contiguous()
		wq := getParam(m, fmt.Sprintf("layers.%d.attention.wq.weight", i)).Transpose(0, 1).Contiguous()
		wv := getParam(m, fmt.Sprintf("layers.%d.attention.wv.weight", i)).Transpose(0, 1).Contiguous()
		wo := getParam(m, fmt.Sprintf("layers.%d.attention.wo.weight", i)).Transpose(0, 1).Contiguous()
		norm1 := getParam(m, fmt.Sprintf("layers.%d.attention_norm.weight", i))
		w1 := getParam(m, fmt.Sprintf("layers.%d.feed_forward.w1.weight", i)).Transpose(0, 1).Contiguous()
		w2 := getParam(m, fmt.Sprintf("layers.%d.feed_forward.w2.weight", i)).Transpose(0, 1).Contiguous()
		w3 := getParam(m, fmt.Sprintf("layers.%d.feed_forward.w3.weight", i)).Transpose(0, 1).Contiguous()
		norm2 := getParam(m, fmt.Sprintf("layers.%d.ffn_norm.weight", i))

		var block block
		block.attn = ilayer.NewAttention(wq, wk, wv, wo, params.Heads, params.Dim)
		block.attnNorm = ilayer.NewRMSNorm(norm1, eps)
		block.ffn = newFeedforward(w1, w2, w3)
		block.ffnNorm = ilayer.NewRMSNorm(norm2, eps)

		md.blocks = append(md.blocks, &block)
	}
	md.norm = ilayer.NewRMSNorm(getParam(m, "norm.weight"), eps)
	md.output = ilayer.NewLinear(getParam(m, "output.weight").Transpose(0, 1).Contiguous())
	params.Vocabs = int(getParam(m, "output.weight").Shapes()[0])
	return &md
}

func LoadFromTNN(dir string, params *Params) *Model {
	var net net.Net
	err := net.Load(dir)
	runtime.Assert(err)
	layers := net.Layers()

	var md Model
	md.prepareFreqs(params.Dim/params.Heads, 2048*2)

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

func (m *Model) prepareFreqs(dim, end int64) {
	const theta = 10000.
	freqs := make([]float32, dim/2)
	for i := 0; i < len(freqs); i++ {
		freqs[i] = 1 / float32(math.Pow(theta, float64(i*2)/float64(dim)))
	}
	f := tensor.FromFloat32(nil, freqs, tensor.WithShapes(dim/2))
	ts := make([]float32, end)
	for i := int64(0); i < end; i++ {
		ts[i] = float32(i)
	}
	t := tensor.FromFloat32(nil, ts, tensor.WithShapes(end))
	freq := tensor.Outer(t, f)
	m.freqs = tensor.Polar(ones(freq.Shapes()), freq)
}

func ones(shapes []int64) *tensor.Tensor {
	size := shapes[0]
	for i := 1; i < len(shapes); i++ {
		size *= shapes[i]
	}
	data := make([]float32, size)
	for i := 0; i < len(data); i++ {
		data[i] = 1
	}
	return tensor.FromFloat32(nil, data, tensor.WithShapes(shapes...))
}

func (m *Model) Forward(x *tensor.Tensor) *tensor.Tensor {
	seqlen := x.Shapes()[1]
	h := m.embedding.Forward(x)
	// fmt.Println(h.NArrow(1, 0, 1).NArrow(2, 0, 16).BFloat16Value())
	// fmt.Println(h.NArrow(1, 1, 1).NArrow(2, 0, 16).BFloat16Value())
	// fmt.Println(h.NArrow(1, 2, 1).NArrow(2, 0, 16).BFloat16Value())
	for _, block := range m.blocks {
		h = block.forward(h, m.freqs.NArrow(0, 0, seqlen))
	}
	h = m.norm.Forward(h)
	output := m.output.Forward(h)
	return output
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
