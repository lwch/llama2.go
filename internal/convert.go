package internal

import (
	"encoding/json"
	"fmt"
	imodel "llama2/internal/model"
	"os"
	"path/filepath"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/model"
	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/lwch/sentencepiece"
	"github.com/spf13/cobra"
)

var ModelDir string
var ModelName string
var OutputDir string

func Convert(*cobra.Command, []string) {
	s := mmgr.New()
	defer s.GC()

	dir := filepath.Join(ModelDir, "tokenizer.model")
	logging.Info("loading tokenizer from %s...", dir)
	tk, err := sentencepiece.Load(dir)
	runtime.Assert(err)

	dir = filepath.Join(ModelDir, ModelName, "params.json")
	logging.Info("loading params from %s...", dir)
	params := imodel.LoadParam(dir)
	params.Vocabs = tk.Count()
	data, _ := json.MarshalIndent(params, "", "  ")
	fmt.Println(string(data))

	dir = filepath.Join(ModelDir, ModelName, "consolidated.00.pth")
	logging.Info("loading model from %s...", dir)
	m, err := model.Load(dir, s)
	runtime.Assert(err)
	logging.Info("model loaded")

	md := imodel.LoadFromTorch(m, params)

	os.MkdirAll(OutputDir, 0755)
	dir = filepath.Join(OutputDir, "llama2.model")
	logging.Info("saving model to %s...", dir)
	md.ToScalarType(consts.KBFloat16).Save(dir) // TODO: quantize
	logging.Info("model saved")
}
