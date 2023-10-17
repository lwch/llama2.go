package model

import (
	"encoding/json"
	"llama2/internal/param"
	"os"

	"github.com/lwch/runtime"
)

type ParamInfo struct {
	Type  param.Type `json:"type"`
	Shape []int64    `json:"shape"`
}

type Params struct {
	Dim        int64                `json:"dim"`
	MultipleOf int                  `json:"multiple_of"`
	Heads      int64                `json:"n_heads"`
	KVHeads    int64                `json:"n_kv_heads,omitempty"`
	Layers     int                  `json:"n_layers"`
	Eps        float32              `json:"norm_eps"`
	Vocabs     int64                `json:"vocab_size"`
	Params     map[string]ParamInfo `json:"params,omitempty"`
}

func LoadParam(dir string) *Params {
	f, err := os.Open(dir)
	runtime.Assert(err)
	defer f.Close()
	var params Params
	runtime.Assert(json.NewDecoder(f).Decode(&params))
	return &params
}
