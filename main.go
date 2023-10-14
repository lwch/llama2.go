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
	Args:  cobra.MinimumNArgs(1),
	Run:   internal.Convert,
}

var textCompletionCmd = &cobra.Command{
	Use:   "text-completion",
	Short: "Text completion",
	Run:   internal.TextCompletion,
}

func main() {
	convertCmd.Flags().StringVar(&internal.OutputDir, "output", "./llama2.model", "output directory")
	rootCmd.AddCommand(convertCmd)

	textCompletionCmd.Flags().StringVar(&internal.ModelDir, "model", "./models", "model directory")
	textCompletionCmd.Flags().BoolVar(&internal.CacheParam, "cache", false, "cache param in memory")
	textCompletionCmd.Flags().IntVar(&internal.MaxInferenceLength, "max-length", 16, "max inference length")
	textCompletionCmd.Flags().Float32Var(&internal.Temperature, "temperature", 0.6, "temperature")
	textCompletionCmd.Flags().Float32Var(&internal.TopP, "top-p", 0.9, "top p")
	runtime.Assert(textCompletionCmd.MarkFlagRequired("model"))
	rootCmd.AddCommand(textCompletionCmd)

	rootCmd.CompletionOptions.DisableDefaultCmd = true
	runtime.Assert(rootCmd.Execute())
}
