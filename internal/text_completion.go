package internal

import (
	"bytes"
	"fmt"
	"io"
	"llama2/internal/model"
	"os"
	"time"

	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

var CacheParam bool
var MaxInferenceLength int

func TextCompletion(*cobra.Command, []string) {
	md := model.Load(ModelDir)
	logging.Info("model loaded")
	md.ShowInfo()

	tk := md.GetTokenizer()

	input, err := io.ReadAll(os.Stdin)
	runtime.Assert(err)
	input = bytes.TrimSpace(input)
	tks := tk.Encode(string(input), true, false)

	ctx := md.NewContext(CacheParam)
	var cursor int64
	var nextToken uint64
	for _, token := range tks {
		begin := time.Now()
		scores, err := md.Forward(ctx, token, cursor)
		cost := time.Since(begin)
		runtime.Assert(err)
		nextToken = getLabel(scores)
		prompt := tk.Decode([]uint64{token})
		if token == uint64(tk.Bos()) {
			prompt = "<s>"
		}
		inference := tk.Decode([]uint64{nextToken})
		fmt.Printf("cost: %s, prompt: %s, inference: %s\n", cost, prompt, inference)
		cursor++
	}

	for i := 0; i < MaxInferenceLength; i++ {
		begin := time.Now()
		scores, err := md.Forward(ctx, nextToken, cursor)
		cost := time.Since(begin)
		runtime.Assert(err)
		nextToken = getLabel(scores)
		inference := tk.Decode([]uint64{nextToken})
		if nextToken == uint64(tk.Eos()) {
			inference = "</s>"
		}
		fmt.Printf("cost: %s, inference: %s\n", cost, inference)
		cursor++
		if nextToken == uint64(tk.Eos()) {
			break
		}
	}
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
