package internal

import (
	"path/filepath"

	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/lwch/sentencepiece"
	"github.com/spf13/cobra"
)

func TextCompletion(*cobra.Command, []string) {
	dir := filepath.Join(ModelDir, "tokenizer.model")
	logging.Info("loading tokenizer from %s...", dir)
	tk, err := sentencepiece.Load(dir)
	runtime.Assert(err)
	_ = tk
}
