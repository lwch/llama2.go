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

	tk := md.GetTokenizer()
	samp := sampler.New(Temperature, TopP)

	ctx := md.NewContext(false)
	var system, user string
	var tks []uint64
	for {
		system = strings.TrimSpace(readStdin("Enter system prompt (optional): "))
		for len(user) == 0 {
			user = strings.TrimSpace(readStdin("Enter user prompt: "))
		}
		fmt.Print("thinking")
		var input string
		if len(system) == 0 {
			input = fmt.Sprintf("[INST] %s [/INST]", user)
		} else {
			input = fmt.Sprintf("[INST] <<SYS>\n%s\n<</SYS>\n\n%s [/INST]", system, user)
		}
		tokens := tk.Encode(input, true, false)
		for _, token := range tokens {
			_, err := md.Forward(ctx, token, int64(len(tks)))
			runtime.Assert(err)
			fmt.Print(".")
			tks = append(tks, token)
		}
		fmt.Println()
		for {
			scores, err := md.Forward(ctx, uint64(tk.Eos()), int64(len(tks)))
			runtime.Assert(err)
			token := samp.Sample(scores)
			fmt.Print(token)
			tks = append(tks, token)
			if token == uint64(tk.Eos()) {
				break
			}
		}
		fmt.Println()
	}
}
