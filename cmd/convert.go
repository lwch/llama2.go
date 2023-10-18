package cmd

import (
	"encoding/json"
	"fmt"
	"llama2/internal/model"
	"llama2/internal/model/checkpoint"
	"llama2/internal/param"
	"os"
	"path/filepath"
	"sort"

	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

var OutputDir string
var Quantize string
var GroupSize int64

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
	t := param.TypeBF16
	switch Quantize {
	case "i8":
		t = param.TypeQI8
	case "u8":
		t = param.TypeQU8
	case "i7":
		t = param.TypeQI7
	case "u7":
		t = param.TypeQU7
	case "i6":
		t = param.TypeQI6
	case "u6":
		t = param.TypeQU6
	case "i5":
		t = param.TypeQI5
	case "u5":
		t = param.TypeQU5
	case "i4":
		t = param.TypeQI4
	case "u4":
		t = param.TypeQU4
	}
	model.Convert(ckpts, params, filepath.Join(modelDir, "..", "tokenizer.model"), OutputDir, t, GroupSize)

	params.Params = nil
	data, _ := json.MarshalIndent(params, "", "  ")
	fmt.Println("params:")
	fmt.Println(string(data))
}
