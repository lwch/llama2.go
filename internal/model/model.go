package model

import (
	"encoding/binary"
	"fmt"
	"llama2/internal/model/checkpoint"
	"llama2/internal/tensor"
	"os"
	"sync"

	"github.com/klauspost/compress/zip"
	"github.com/lwch/logging"
	"github.com/lwch/runtime"
)

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

func LoadFromCheckpoint(ckpt *checkpoint.Model, params *Params) *Model {
	getParam := func(name string) tensor.Tensor {
		t := ckpt.Params()[name]
		if t == nil {
			panic(fmt.Sprintf("cannot find %s", name))
		}
		return tensor.NewBFloat16(t.GetShape(), t.Get().([]uint16))
	}

	transpose := func(t tensor.Tensor) tensor.Tensor {
		rows, cols := t.Shapes()[0], t.Shapes()[1]
		data := t.Raw().([]uint16)
		for i := int64(0); i < rows; i++ {
			for j := int64(0); j < cols; j++ {
				data[i*cols+j], data[j*rows+i] = data[j*rows+i], data[i*cols+j]
			}
		}
		t.Shapes()[0], t.Shapes()[1] = cols, rows
		return t
	}

	var md Model
	md.embeddingWeight = getParam("tok_embeddings.weight")
	var wg sync.WaitGroup
	wg.Add(params.Layers)
	md.attentionWQ = make([]tensor.Tensor, params.Layers)
	md.attentionWK = make([]tensor.Tensor, params.Layers)
	md.attentionWV = make([]tensor.Tensor, params.Layers)
	md.attentionWO = make([]tensor.Tensor, params.Layers)
	md.attentionNorm = make([]tensor.Tensor, params.Layers)
	md.ffnW1 = make([]tensor.Tensor, params.Layers)
	md.ffnW2 = make([]tensor.Tensor, params.Layers)
	md.ffnW3 = make([]tensor.Tensor, params.Layers)
	md.ffnNorm = make([]tensor.Tensor, params.Layers)
	for i := 0; i < params.Layers; i++ {
		go func(i int) {
			defer wg.Done()
			wq := getParam(fmt.Sprintf("layers.%d.attention.wq.weight", i))
			wk := getParam(fmt.Sprintf("layers.%d.attention.wk.weight", i))
			wv := getParam(fmt.Sprintf("layers.%d.attention.wv.weight", i))
			wo := getParam(fmt.Sprintf("layers.%d.attention.wo.weight", i))
			md.attentionWQ[i] = transpose(wq)
			md.attentionWK[i] = transpose(wk)
			md.attentionWV[i] = transpose(wv)
			md.attentionWO[i] = transpose(wo)
			md.attentionNorm[i] = getParam(fmt.Sprintf("layers.%d.attention_norm.weight", i))
			w1 := getParam(fmt.Sprintf("layers.%d.feed_forward.w1.weight", i))
			w2 := getParam(fmt.Sprintf("layers.%d.feed_forward.w2.weight", i))
			w3 := getParam(fmt.Sprintf("layers.%d.feed_forward.w3.weight", i))
			md.ffnW1[i] = transpose(w1)
			md.ffnW2[i] = transpose(w2)
			md.ffnW3[i] = transpose(w3)
			md.ffnNorm[i] = getParam(fmt.Sprintf("layers.%d.ffn_norm.weight", i))
			logging.Info("  - layer %d loaded", i)
		}(i)
	}
	wg.Wait()
	md.norm = getParam("norm.weight")
	md.output = transpose(getParam("output.weight"))
	params.Vocabs = md.embeddingWeight.Shapes()[0]
	return &md
}

func (m *Model) Save(dir string) {
	f, err := os.Create(dir)
	runtime.Assert(err)
	defer f.Close()
	zw := zip.NewWriter(f)
	defer zw.Close()
	write := func(t tensor.Tensor, name string) {
		logging.Info("  => writing %s...", name)
		w, err := zw.CreateHeader(&zip.FileHeader{
			Name:   name,
			Method: zip.Store,
		})
		runtime.Assert(err)
		err = binary.Write(w, binary.LittleEndian, t.BinaryData())
		runtime.Assert(err)
	}
	write(m.embeddingWeight, "embedding_weight")
	for i := 0; i < len(m.attentionWQ); i++ {
		write(m.attentionWQ[i], fmt.Sprintf("attention_%d_wq", i))
		write(m.attentionWK[i], fmt.Sprintf("attention_%d_wk", i))
		write(m.attentionWV[i], fmt.Sprintf("attention_%d_wv", i))
		write(m.attentionWO[i], fmt.Sprintf("attention_%d_wo", i))
		write(m.attentionNorm[i], fmt.Sprintf("attention_%d_norm", i))
		write(m.ffnW1[i], fmt.Sprintf("ffn_%d_w1", i))
		write(m.ffnW2[i], fmt.Sprintf("ffn_%d_w2", i))
		write(m.ffnW3[i], fmt.Sprintf("ffn_%d_w3", i))
		write(m.ffnNorm[i], fmt.Sprintf("ffn_%d_nrom", i))
	}
	write(m.norm, "norm")
	write(m.output, "output")
}
