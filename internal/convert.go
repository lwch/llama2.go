package internal

import (
	"encoding/json"
	"fmt"
	"io"
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

	md := model.LoadFromCheckpoint(ckpt, params)
	data, _ := json.MarshalIndent(params, "", "  ")
	fmt.Println("params:")
	fmt.Println(string(data))

	os.MkdirAll(OutputDir, 0755)
	dir = filepath.Join(OutputDir, "llama2.model")
	logging.Info("saving model to %s...", dir)
	md.Save(dir) // TODO: quantize
	logging.Info("model saved")

	dir = filepath.Join(OutputDir, "params.json")
	logging.Info("saving params to %s...", dir)
	data, err = json.Marshal(params)
	runtime.Assert(err)
	runtime.Assert(os.WriteFile(dir, data, 0644))
	logging.Info("params saved")

	dir = filepath.Join(ModelDir, "tokenizer.model")
	logging.Info("copying tokenizer from %s...", dir)
	runtime.Assert(copyFile(dir, filepath.Join(OutputDir, "tokenizer.model")))
	logging.Info("tokenizer copied")
}

func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()
	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()
	_, err = io.Copy(dstFile, srcFile)
	return err
}
