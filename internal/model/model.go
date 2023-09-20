package model

import (
	"github.com/lwch/gotorch/consts"
	gmodel "github.com/lwch/gotorch/model"
	"github.com/lwch/tnn/nn/net"
)

type Model struct {
	embedding *embeddingLayer
}

func LoadFromTorch(m *gmodel.Model, params *Params) *Model {
	return &Model{
		embedding: newEmbeddingLayer(m.Get("tok_embeddings")),
	}
}

func (m *Model) To(t consts.ScalarType) *Model {
	var model Model
	return &model
}

func (m *Model) Save(dir string) error {
	var net net.Net
	return net.Save(dir)
}
