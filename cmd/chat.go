package cmd

import (
	"bufio"
	"fmt"
	"llama2/internal/model"
	"llama2/internal/sampler"
	"os"
	"strings"

	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

func readStdin(guide string) string {
	fmt.Print(guide)
	s := bufio.NewScanner(os.Stdin)
	s.Scan()
	return s.Text()
}

func Chat(*cobra.Command, []string) {
	md := model.Load(ModelDir)
	logging.Info("model loaded")
	md.ShowInfo()

	if CacheParam {
		logging.Info("warm up model...")
		md.WarmUP(FP32)
		logging.Info("warm up done")
	}

	tk := md.GetTokenizer()
	samp := sampler.New(Temperature, TopP)

	ctx := md.NewContext(CacheParam, FP32)
	var system, user string
	var nextToken int64
	var tks []uint64
	nextToken = -1
	for {
		if len(tks) == 0 {
			system = strings.TrimSpace(readStdin("Enter system prompt (optional): "))
		}
		for len(user) == 0 {
			user = strings.TrimSpace(readStdin("Enter user prompt: "))
		}
		fmt.Print("thinking")
		var input string
		if len(system) == 0 {
			input = fmt.Sprintf("[INST] %s [/INST]", user)
		} else {
			input = fmt.Sprintf("[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", system, user)
		}
		system = ""
		user = ""
		tokens := tk.Encode(input, true, false)
		for i, token := range tokens {
			scores, err := md.Forward(ctx, token, int64(len(tks)))
			runtime.Assert(err)
			fmt.Print(".")
			tks = append(tks, token)
			if i == len(tokens)-1 {
				nextToken = int64(samp.Sample(scores))
			}
		}
		fmt.Println()
		fmt.Print(tk.Decode([]uint64{uint64(nextToken)}))
		for {
			scores, err := md.Forward(ctx, uint64(nextToken), int64(len(tks)))
			runtime.Assert(err)
			nextToken = int64(samp.Sample(scores))
			fmt.Print(tk.Decode([]uint64{uint64(nextToken)}))
			tks = append(tks, uint64(nextToken))
			if nextToken == tk.Eos() {
				break
			}
		}
		md.Forward(ctx, uint64(nextToken), int64(len(tks)))
		fmt.Println()
	}
}
