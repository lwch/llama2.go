package main

import (
	"llama2/cmd"
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
	Run:   cmd.Convert,
}

var textCompletionCmd = &cobra.Command{
	Use:   "text-completion",
	Short: "Text completion",
	Run:   cmd.TextCompletion,
}

var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Chat",
	Run:   cmd.Chat,
}

func main() {
	convertCmd.Flags().StringVarP(&cmd.OutputDir, "output", "o", "./llama2.model", "output directory")
	convertCmd.Flags().StringVarP(&cmd.Quantize, "quantize", "q", "bf16", "bf16, i8, u8, i7, u7, i6, u6, i5, u5, i4, u4")
	convertCmd.Flags().Int64VarP(&cmd.GroupSize, "group", "g", 4*1024, "quantize group size")
	rootCmd.AddCommand(convertCmd)

	textCompletionCmd.Flags().StringVarP(&cmd.ModelDir, "model", "m", "./models", "model directory")
	textCompletionCmd.Flags().BoolVar(&cmd.CacheParam, "cache", false, "cache param in memory")
	textCompletionCmd.Flags().IntVar(&cmd.MaxInferenceLength, "max-length", 16, "max inference length")
	textCompletionCmd.Flags().Float32VarP(&cmd.Temperature, "temperature", "t", 0.6, "temperature")
	textCompletionCmd.Flags().Float32VarP(&cmd.TopP, "top-p", "p", 0.9, "top p")
	runtime.Assert(textCompletionCmd.MarkFlagRequired("model"))
	rootCmd.AddCommand(textCompletionCmd)

	chatCmd.Flags().StringVarP(&cmd.ModelDir, "model", "m", "./models", "model directory")
	chatCmd.Flags().BoolVar(&cmd.CacheParam, "cache", false, "cache param in memory")
	chatCmd.Flags().Float32VarP(&cmd.Temperature, "temperature", "t", 0.6, "temperature")
	chatCmd.Flags().Float32VarP(&cmd.TopP, "top-p", "p", 0.9, "top p")
	runtime.Assert(chatCmd.MarkFlagRequired("model"))
	rootCmd.AddCommand(chatCmd)

	rootCmd.CompletionOptions.DisableDefaultCmd = true
	runtime.Assert(rootCmd.Execute())
}
