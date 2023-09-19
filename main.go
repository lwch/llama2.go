package main

import (
	"llama2/internal"
	"os"

	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:  "llama2",
	Long: "LLaMA model",
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
		os.Exit(1)
	},
}

var textCompletionCmd = &cobra.Command{
	Use:   "text-completion",
	Short: "Text completion",
	Run:   internal.TextCompletion,
}

func main() {
	textCompletionCmd.Flags().StringVar(&internal.ModelDir, "model", "", "model directory")
	textCompletionCmd.Flags().StringVar(&internal.ModelName, "name", "", "model name, like llama-2-7b")
	runtime.Assert(textCompletionCmd.MarkFlagRequired("model"))
	runtime.Assert(textCompletionCmd.MarkFlagRequired("name"))
	rootCmd.AddCommand(textCompletionCmd)

	rootCmd.CompletionOptions.DisableDefaultCmd = true
	runtime.Assert(rootCmd.Execute())
}
