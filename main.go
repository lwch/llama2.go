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
	convertCmd.Flags().StringVar(&internal.ModelName, "name", "", "model name, like llama-2-7b")
	convertCmd.Flags().StringVar(&internal.OutputDir, "output", "./llama2.model", "output directory")
	runtime.Assert(convertCmd.MarkFlagRequired("model"))
	//runtime.Assert(convertCmd.MarkFlagRequired("name"))
	rootCmd.AddCommand(convertCmd)

	textCompletionCmd.Flags().StringVar(&internal.ModelDir, "model", "./models", "model directory")
	textCompletionCmd.Flags().BoolVar(&internal.CacheParam, "cache-param", false, "cache param")
	textCompletionCmd.Flags().IntVar(&internal.MaxInferenceLength, "max-inference-length", 256, "max inference length")
	textCompletionCmd.Flags().Float32Var(&internal.Temperature, "temperature", 0.6, "temperature")
	textCompletionCmd.Flags().Float32Var(&internal.TopP, "top-p", 0.9, "top p")
	runtime.Assert(textCompletionCmd.MarkFlagRequired("model"))
	rootCmd.AddCommand(textCompletionCmd)

	rootCmd.CompletionOptions.DisableDefaultCmd = true
	runtime.Assert(rootCmd.Execute())
}
