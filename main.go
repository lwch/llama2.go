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

var convertCmd = &cobra.Command{
	Use:   "convert",
	Short: "Convert model to tnn model",
	Run:   internal.Convert,
}

var textCompletionCmd = &cobra.Command{
	Use:   "text-completion",
	Short: "Text completion",
	Run:   internal.TextCompletion,
}

func main() {
	convertCmd.Flags().StringVar(&internal.ModelDir, "model", "", "model directory")
	convertCmd.Flags().StringVar(&internal.OutputDir, "output", "", "output directory")
	runtime.Assert(convertCmd.MarkFlagRequired("model"))
	runtime.Assert(convertCmd.MarkFlagRequired("output"))
	rootCmd.AddCommand(convertCmd)

	textCompletionCmd.Flags().StringVar(&internal.ModelDir, "model", "", "model directory")
	textCompletionCmd.Flags().StringVar(&internal.ModelName, "name", "", "model name, like llama-2-7b")
	runtime.Assert(textCompletionCmd.MarkFlagRequired("model"))
	runtime.Assert(textCompletionCmd.MarkFlagRequired("name"))
	rootCmd.AddCommand(textCompletionCmd)

	rootCmd.CompletionOptions.DisableDefaultCmd = true
	runtime.Assert(rootCmd.Execute())
}
