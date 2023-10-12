package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"llama2/internal/model/checkpoint"
	"llama2/internal/param"
	"os"
	"path/filepath"

	"github.com/klauspost/compress/zip"
	"github.com/lwch/logging"
	"github.com/lwch/runtime"
)

func Convert(ckpt *checkpoint.Model, params *Params, tokenizer, output string) {
	params.Vocabs = ckpt.Params()["tok_embeddings.weight"].GetShape()[0]

	f, err := os.Create(output)
	runtime.Assert(err)
	defer f.Close()

	zw := zip.NewWriter(f)
	defer zw.Close()

	writeTokenizer(zw, tokenizer)

	list := ckpt.Params()

	params.Params = make(map[string]ParamInfo)
	params.Params["embedding_weight"] = writeParam(zw, list["tok_embeddings.weight"], "embedding_weight", false)
	for i := 0; i < params.Layers; i++ {
		key := fmt.Sprintf("layers_%d_attention_wq", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.attention.wq.weight", i)], key, true)
		key = fmt.Sprintf("layers_%d_attention_wk", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.attention.wk.weight", i)], key, true)
		key = fmt.Sprintf("layers_%d_attention_wv", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.attention.wv.weight", i)], key, true)
		key = fmt.Sprintf("layers_%d_attention_wo", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.attention.wo.weight", i)], key, true)

		key = fmt.Sprintf("layers_%d_attention_norm", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.attention_norm.weight", i)], key, false)

		key = fmt.Sprintf("layers_%d_ffn_w1", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.feed_forward.w1.weight", i)], key, true)
		key = fmt.Sprintf("layers_%d_ffn_w2", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.feed_forward.w2.weight", i)], key, true)
		key = fmt.Sprintf("layers_%d_ffn_w3", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.feed_forward.w3.weight", i)], key, true)

		key = fmt.Sprintf("layers_%d_ffn_norm", i)
		params.Params[key] = writeParam(zw, list[fmt.Sprintf("layers.%d.ffn_norm.weight", i)], key, false)
	}
	params.Params["norm"] = writeParam(zw, list["norm.weight"], "norm", false)
	params.Params["output"] = writeParam(zw, list["output.weight"], "output", true)

	writeParams(zw, params)
}

func writeParams(zw *zip.Writer, params *Params) {
	logging.Info("  => writing params.json...")
	f, err := zw.Create("params.json")
	runtime.Assert(err)
	runtime.Assert(json.NewEncoder(f).Encode(params))
}

func writeTokenizer(zw *zip.Writer, tokenizer string) {
	logging.Info("  => writing tokenizer.model from %s...", tokenizer)
	f, err := zw.Create("tokenizer.model")
	runtime.Assert(err)
	src, err := os.Open(tokenizer)
	runtime.Assert(err)
	defer src.Close()
	_, err = io.Copy(f, src)
	runtime.Assert(err)
}

// TODO: quantization
func writeParam(zw *zip.Writer, p checkpoint.Storage, name string, transpose bool) ParamInfo {
	logging.Info("  => loading param %s...", name)
	dt, err := p.Load()
	runtime.Assert(err)
	data := dt.([]uint16)
	shapes := p.GetShape()
	logging.Info("  => param %s shape: %v", name, shapes)
	if transpose {
		if len(shapes) != 2 {
			panic("len(shapes)!=2")
		}
		cols, rows := shapes[0], shapes[1]
		t := make([]uint16, rows*cols)
		for i := int64(0); i < rows; i++ {
			for j := int64(0); j < cols; j++ {
				t[i*cols+j] = data[j*rows+i]
			}
		}
		data = t
		shapes[0], shapes[1] = shapes[1], shapes[0]
		logging.Info("  => param %s transpose shape to %v", name, shapes)
	}
	w, err := zw.CreateHeader(&zip.FileHeader{
		Name:   filepath.Join("params", name),
		Method: zip.Store,
	})
	runtime.Assert(err)
	runtime.Assert(binary.Write(w, binary.LittleEndian, data))
	return ParamInfo{
		Type:  param.TypeBF16,
		Shape: shapes,
	}
}
