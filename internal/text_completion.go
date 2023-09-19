package internal

import (
	"path/filepath"

	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/model"
	"github.com/lwch/runtime"
	"github.com/spf13/cobra"
)

var ModelDir string
var ModelName string

func TextCompletion(*cobra.Command, []string) {
	s := mmgr.New()
	dir := filepath.Join(ModelDir, ModelName, "consolidated.00.pth")
	m, err := model.Load(dir, s)
	runtime.Assert(err)
	_ = m
	select {}
}
