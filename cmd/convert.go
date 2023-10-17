package cmd

import (
	"encoding/json"
	"fmt"
	"llama2/internal/model"
	"llama2/internal/model/checkpoint"
	"os"
	"path/filepath"
	"sort"

	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

var OutputDir string

func Convert(cmd *cobra.Command, args []string) {
	modelDir := args[0]
	dir := filepath.Join(modelDir, "params.json")
	logging.Info("loading params from %s...", dir)
	params := model.LoadParam(dir)

	files, err := filepath.Glob(filepath.Join(modelDir, "consolidated.*.pth"))
	runtime.Assert(err)
	sort.Strings(files)
	var ckpts []*checkpoint.Model
	for _, file := range files {
		ckpt, err := checkpoint.Load(file)
		runtime.Assert(err)
		logging.Info("checkpoint %s loaded", file)
		ckpts = append(ckpts, ckpt)
	}

	os.MkdirAll(filepath.Dir(OutputDir), 0755)
	logging.Info("convert to llama2 model...")
	model.Convert(ckpts, params, filepath.Join(modelDir, "..", "tokenizer.model"), OutputDir)

	params.Params = nil
	data, _ := json.MarshalIndent(params, "", "  ")
	fmt.Println("params:")
	fmt.Println(string(data))
}
