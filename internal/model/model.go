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
	var md Model
	md.embedding = newEmbeddingLayer(m.Get("tok_embeddings.weight"))
	return &md
}

func (m *Model) ToScalarType(t consts.ScalarType) *Model {
	var model Model
	model.embedding = m.embedding.ToScalarType(t)
	return &model
}

func (m *Model) Save(dir string) error {
	var net net.Net
	net.Add(m.embedding)
	return net.Save(dir)
}
