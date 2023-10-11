package internal

import (
	"encoding/json"
	"fmt"
	"llama2/internal/model"
	"llama2/internal/model/checkpoint"
	"os"
	"path/filepath"

	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

var ModelDir string
var ModelName string
var OutputDir string

func Convert(*cobra.Command, []string) {
	dir := filepath.Join(ModelDir, ModelName, "params.json")
	logging.Info("loading params from %s...", dir)
	params := model.LoadParam(dir)

	dir = filepath.Join(ModelDir, ModelName, "consolidated.00.pth")
	logging.Info("loading checkpoint from %s...", dir)
	ckpt, err := checkpoint.Load(dir)
	runtime.Assert(err)
	logging.Info("checkpoint loaded")

	os.MkdirAll(filepath.Dir(OutputDir), 0755)
	logging.Info("convert to llama2 model...")
	model.Convert(ckpt, params, filepath.Join(ModelDir, "tokenizer.model"), OutputDir)

	params.Params = nil
	data, _ := json.MarshalIndent(params, "", "  ")
	fmt.Println("params:")
	fmt.Println(string(data))
}
