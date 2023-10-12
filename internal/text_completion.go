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

func TextCompletion(*cobra.Command, []string) {
	md := model.Load(ModelDir)
	logging.Info("model loaded")
	md.ShowInfo()

	tk := md.GetTokenizer()

	input, err := io.ReadAll(os.Stdin)
	runtime.Assert(err)
	input = bytes.TrimSpace(input)
	tks := tk.Encode(string(input), true, false)

	ctx := md.NewContext()
	var nextToken uint64
	for i, token := range tks {
		begin := time.Now()
		scores, err := md.Forward(ctx, token, int64(i))
		runtime.Assert(err)
		nextToken = getLabel(scores)
		str := tk.Decode([]uint64{token, nextToken})
		fmt.Println(time.Since(begin), str)
	}

	for {
		begin := time.Now()
		scores, err := md.Forward(ctx, nextToken, int64(len(tks)))
		runtime.Assert(err)
		nextToken = getLabel(scores)
		str := tk.Decode([]uint64{nextToken})
		fmt.Println(time.Since(begin), str)
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
