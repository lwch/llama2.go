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

func Convert(ckpts []*checkpoint.Model, params *Params, tokenizer, output string) {
	params.Vocabs = ckpts[0].Params()["tok_embeddings.weight"].Shape()[0]

	f, err := os.Create(output)
	runtime.Assert(err)
	defer f.Close()

	zw := zip.NewWriter(f)
	defer zw.Close()

	writeTokenizer(zw, tokenizer)

	params.Params = make(map[string]ParamInfo)
	key := "tok_embeddings.weight"
	params.Params[key] = writeParam(ckpts, zw, key, true)
	for i := int64(0); i < params.Layers; i++ {
		key = fmt.Sprintf("layers.%d.attention.wq.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, false)
		key = fmt.Sprintf("layers.%d.attention.wk.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, false)
		key = fmt.Sprintf("layers.%d.attention.wv.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, false)
		key = fmt.Sprintf("layers.%d.attention.wo.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, true)

		key = fmt.Sprintf("layers.%d.attention_norm.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, false)

		key = fmt.Sprintf("layers.%d.feed_forward.w1.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, false)
		key = fmt.Sprintf("layers.%d.feed_forward.w2.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, true)
		key = fmt.Sprintf("layers.%d.feed_forward.w3.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, false)

		key = fmt.Sprintf("layers.%d.ffn_norm.weight", i)
		params.Params[key] = writeParam(ckpts, zw, key, false)
	}
	key = "norm.weight"
	params.Params[key] = writeParam(ckpts, zw, key, false)
	key = "output.weight"
	params.Params[key] = writeParam(ckpts, zw, key, false)

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

func getParamData(ckpts []*checkpoint.Model, name string, rowParallel bool) ([]uint16, []int64, error) {
	shape := ckpts[0].Params()[name].Shape()
	if len(shape) == 1 {
		p, err := ckpts[0].Params()[name].Load()
		if err != nil {
			return nil, nil, err
		}
		return p.([]uint16), shape, nil
	}
	if rowParallel {
		datas := make([][]uint16, len(ckpts))
		for i, ckpt := range ckpts {
			p, err := ckpt.Params()[name].Load()
			if err != nil {
				return nil, nil, err
			}
			datas[i] = p.([]uint16)
		}
		data := make([]uint16, shape[0]*shape[1]*int64(len(ckpts)))
		for y := int64(0); y < shape[0]; y++ {
			for i := 0; i < len(datas); i++ {
				idx := y*shape[1]*int64(len(datas)) + int64(i)*shape[1]
				copy(data[idx:idx+shape[1]], datas[i][y*shape[1]:(y+1)*shape[1]])
			}
		}
		shape[1] *= int64(len(ckpts))
		return data, shape, nil
	}
	var data []uint16
	for _, ckpt := range ckpts {
		dt, err := ckpt.Params()[name].Load()
		if err != nil {
			return nil, nil, err
		}
		data = append(data, dt.([]uint16)...)
	}
	shape[0] *= int64(len(ckpts))
	return data, shape, nil
}

// TODO: quantization
func writeParam(ckpts []*checkpoint.Model, zw *zip.Writer, name string, rowParallel bool) ParamInfo {
	logging.Info("  => loading param %s...", name)
	data, shapes, err := getParamData(ckpts, name, rowParallel)
	runtime.Assert(err)
	logging.Info("  => param %s shape: %v", name, shapes)
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
