package internal

import (
	"fmt"
	imodel "llama2/internal/model"
	"path/filepath"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/model"
	"github.com/lwch/logging"
	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

var ModelDir string
var OutputDir string

func Convert(*cobra.Command, []string) {
	s := mmgr.New()
	defer s.GC()

	dir := filepath.Join(ModelDir, "consolidated.00.pth")
	logging.Info("loading model from %s...", dir)
	m, err := model.Load(dir, s)
	runtime.Assert(err)
	logging.Info("model loaded")

	fmt.Println(m.Params())

	dir = filepath.Join(ModelDir, "params.json")
	logging.Info("loading params from %s...", dir)
	params := imodel.LoadParam(dir)
	logging.Info("params loaded")

	md := imodel.LoadFromTorch(m, params)
	md.To(consts.KBFloat16).Save(OutputDir)
}
