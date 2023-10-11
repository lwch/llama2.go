package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"llama2/internal/param"
	"os"
	"path/filepath"

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

	key := "embedding_weight"
	info, ok := params.Params[key]
	if !ok {
		panic(errors.New("embedding_weight info not found"))
	}
	md.embeddingWeight = loadParam(dir, zr, "embedding_weight", info)

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
			key := fmt.Sprintf("layers_%d_%s", i, name)
			info, ok := params.Params[key]
			if !ok {
				panic(fmt.Errorf("%s info not found", key))
			}
			return loadParam(dir, zr, key, info)
		}
		md.attentionWQ = append(md.attentionWQ, loadLayerParam("attention_wq"))
		md.attentionWK = append(md.attentionWK, loadLayerParam("attention_wk"))
		md.attentionWV = append(md.attentionWV, loadLayerParam("attention_wv"))
		md.attentionWO = append(md.attentionWO, loadLayerParam("attention_wo"))
		md.attentionNorm = append(md.attentionNorm, loadLayerParam("attention_norm"))
		md.ffnW1 = append(md.ffnW1, loadLayerParam("ffn_w1"))
		md.ffnW2 = append(md.ffnW2, loadLayerParam("ffn_w2"))
		md.ffnW3 = append(md.ffnW3, loadLayerParam("ffn_w3"))
		md.ffnNorm = append(md.ffnNorm, loadLayerParam("ffn_norm"))
	}
	md.norm = loadParam(dir, zr, "norm", params.Params["norm"])
	md.output = loadParam(dir, zr, "output", params.Params["output"])

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
