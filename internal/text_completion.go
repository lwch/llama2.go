package internal

import (
	"bytes"
	"fmt"
	"io"
	"llama2/internal/model"
	"os"
	"path/filepath"
	rt "runtime"

	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/lwch/sentencepiece"
	"github.com/spf13/cobra"
)

func TextCompletion(*cobra.Command, []string) {
	s := mmgr.New()
	defer s.GC()

	dir := filepath.Join(ModelDir, "tokenizer.model")
	logging.Info("loading tokenizer from %s...", dir)
	tk, err := sentencepiece.Load(dir)
	runtime.Assert(err)
	logging.Info("tokenizer model loaded, token size: %d", tk.Count())

	dir = filepath.Join(ModelDir, "params.json")
	logging.Info("loading params from %s...", dir)
	params := model.LoadParam(dir)

	dir = filepath.Join(ModelDir, "llama2.model")
	logging.Info("loading model from %s...", dir)
	m := model.LoadFromTNN(dir, params)
	logging.Info("model loaded")

	rt.GC()

	input, err := io.ReadAll(os.Stdin)
	runtime.Assert(err)
	input = bytes.TrimSpace(input)
	tks := tk.Encode(string(input), true, false)
	fmt.Print(string(input))
	for {
		data := m.Forward(buildInput(s, tks)).
			NArrow(1, -1, 1).View(-1).
			Softmax(-1).BFloat16Value()
		label := getLabel(data)
		fmt.Print(tk.Decode([]uint64{label}))
		s.GC()
		rt.GC()
		if label == uint64(tk.Eos()) {
			fmt.Println()
			break
		}
		tks = append(tks, label)
	}
}

func buildInput(s *mmgr.Storage, tks []uint64) *tensor.Tensor {
	data := make([]int64, len(tks))
	for i, v := range tks {
		data[i] = int64(v)
	}
	return tensor.FromInt64(s, data,
		tensor.WithShapes(1, int64(len(tks))))
}

func getLabel(data []float32) uint64 {
	var idx uint64
	var max float32
	for i, v := range data {
		if v > max {
			max = v
			idx = uint64(i)
		}
	}
	return idx
}
