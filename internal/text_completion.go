package internal

import (
	"bytes"
	"fmt"
	"io"
	"llama2/internal/model"
	"llama2/internal/sampler"
	"os"
	"time"

	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

var CacheParam bool
var MaxInferenceLength int
var Temperature float32
var TopP float32

func TextCompletion(*cobra.Command, []string) {
	md := model.Load(ModelDir)
	logging.Info("model loaded")
	md.ShowInfo()

	if CacheParam {
		logging.Info("warm up model...")
		md.WarmUP()
		logging.Info("warm up done")
	}

	tk := md.GetTokenizer()

	input, err := io.ReadAll(os.Stdin)
	runtime.Assert(err)
	input = bytes.TrimSpace(input)
	tks := tk.Encode(string(input), true, false)

	samp := sampler.New(Temperature, TopP)

	ctx := md.NewContext(CacheParam)
	var cursor int64
	var nextToken uint64
	for _, token := range tks {
		begin := time.Now()
		scores, err := md.Forward(ctx, token, cursor)
		cost := time.Since(begin)
		runtime.Assert(err)
		nextToken = samp.Sample(scores)
		prompt := tk.Decode([]uint64{token})
		if token == uint64(tk.Bos()) {
			prompt = "<s>"
		}
		inference := tk.Decode([]uint64{nextToken})
		fmt.Printf("cost: %s, prompt: [%s], inference: [%s]\n", cost, prompt, inference)
		cursor++
	}

	for i := 0; i < MaxInferenceLength; i++ {
		begin := time.Now()
		scores, err := md.Forward(ctx, nextToken, cursor)
		cost := time.Since(begin)
		runtime.Assert(err)
		nextToken = samp.Sample(scores)
		inference := tk.Decode([]uint64{nextToken})
		if nextToken == uint64(tk.Eos()) {
			inference = "</s>"
		}
		fmt.Printf("cost: %s, inference: [%s]\n", cost, inference)
		cursor++
		if nextToken == uint64(tk.Eos()) {
			break
		}
	}
}
