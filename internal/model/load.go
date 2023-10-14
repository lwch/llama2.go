package model

import (
	"encoding/json"
	"fmt"
	"llama2/internal/param"
	"os"
	"path/filepath"
	rt "runtime"
	"sync"

	"github.com/klauspost/compress/zip"
	"github.com/lwch/runtime"
	"github.com/lwch/sentencepiece"
)

func Load(dir string) *Model {
	f, err := os.Open(dir)
	runtime.Assert(err)
	defer f.Close()

	fi, err := f.Stat()
	runtime.Assert(err)
	zr, err := zip.NewReader(f, fi.Size())
	runtime.Assert(err)

	params := loadParams(zr)

	var md Model
	md.embeddingDim = params.Dim
	md.layers = params.Layers
	md.heads = params.Heads
	md.eps = params.Eps
	md.tk = loadTokenizer(zr)

	key := "tok_embeddings.weight"
	info, ok := params.Params[key]
	if !ok {
		panic(fmt.Errorf("%s info not found", key))
	}
	md.embeddingWeight = loadParam(dir, zr, key, info)

	md.attentionWQ = nil
	md.attentionWK = nil
	md.attentionWV = nil
	md.attentionWO = nil
	md.attentionNorm = nil
	md.ffnW1 = nil
	md.ffnW2 = nil
	md.ffnW3 = nil
	md.ffnNorm = nil
	for i := 0; i < params.Layers; i++ {
		loadLayerParam := func(name string) param.Param {
			key := fmt.Sprintf("layers.%d.%s", i, name)
			info, ok := params.Params[key]
			if !ok {
				panic(fmt.Errorf("%s info not found", key))
			}
			return loadParam(dir, zr, key, info)
		}
		md.attentionWQ = append(md.attentionWQ, loadLayerParam("attention.wq.weight"))
		md.attentionWK = append(md.attentionWK, loadLayerParam("attention.wk.weight"))
		md.attentionWV = append(md.attentionWV, loadLayerParam("attention.wv.weight"))
		md.attentionWO = append(md.attentionWO, loadLayerParam("attention.wo.weight"))
		md.attentionNorm = append(md.attentionNorm, loadLayerParam("attention_norm.weight"))
		md.ffnW1 = append(md.ffnW1, loadLayerParam("feed_forward.w1.weight"))
		md.ffnW2 = append(md.ffnW2, loadLayerParam("feed_forward.w2.weight"))
		md.ffnW3 = append(md.ffnW3, loadLayerParam("feed_forward.w3.weight"))
		md.ffnNorm = append(md.ffnNorm, loadLayerParam("ffn_norm.weight"))
	}
	key = "norm.weight"
	info, ok = params.Params[key]
	if !ok {
		panic(fmt.Errorf("%s info not found", key))
	}
	md.norm = loadParam(dir, zr, key, info)

	key = "output.weight"
	info, ok = params.Params[key]
	if !ok {
		panic(fmt.Errorf("%s info not found", key))
	}
	md.output = loadParam(dir, zr, key, info)

	return &md
}

func loadParams(zr *zip.Reader) *Params {
	f, err := zr.Open("params.json")
	runtime.Assert(err)
	defer f.Close()
	var ret Params
	runtime.Assert(json.NewDecoder(f).Decode(&ret))
	return &ret
}

func loadParam(modelDir string, zr *zip.Reader, name string, info ParamInfo) param.Param {
	name = filepath.Join("params", name)
	f, err := zr.Open(name)
	runtime.Assert(err)
	defer f.Close()
	switch info.Type {
	case param.TypeBF16:
		return param.NewBF16(modelDir, name, info.Shape)
	default: // TODO: load quantized param
		return nil
	}
}

func loadTokenizer(zr *zip.Reader) *sentencepiece.Model {
	f, err := zr.Open("tokenizer.model")
	runtime.Assert(err)
	defer f.Close()
	tokenizer, err := sentencepiece.LoadFrom(f)
	runtime.Assert(err)
	return tokenizer
}

func (m *Model) WarmUP() {
	var wg sync.WaitGroup
	load := func(p param.Param) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.Load(true)
		}()
	}
	load(m.embeddingWeight)
	for _, p := range m.attentionWQ {
		load(p)
	}
	for _, p := range m.attentionWK {
		load(p)
	}
	for _, p := range m.attentionWV {
		load(p)
	}
	for _, p := range m.attentionWO {
		load(p)
	}
	for _, p := range m.attentionNorm {
		load(p)
	}
	for _, p := range m.ffnW1 {
		load(p)
	}
	for _, p := range m.ffnW2 {
		load(p)
	}
	for _, p := range m.ffnW3 {
		load(p)
	}
	for _, p := range m.ffnNorm {
		load(p)
	}
	load(m.norm)
	load(m.output)
	wg.Wait()
	rt.GC()
}
