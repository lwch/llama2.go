package model

import (
	"encoding/json"
	"os"

	"github.com/lwch/runtime"
)

type Params struct {
	Dim        int64   `json:"dim"`
	MultipleOf int     `json:"multiple_of"`
	Heads      int64   `json:"n_heads"`
	Layers     int     `json:"n_layers"`
	Eps        float32 `json:"norm_eps"`
	Vocabs     int64   `json:"vocab_size"`
}

func LoadParam(dir string) *Params {
	f, err := os.Open(dir)
	runtime.Assert(err)
	defer f.Close()
	var params Params
	runtime.Assert(json.NewDecoder(f).Decode(&params))
	return &params
}
