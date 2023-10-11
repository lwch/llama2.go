package model

import (
	"fmt"
)

func (m *Model) LookupEmbedding(tks []uint64) ([]float32, error) {
	ret := make([]float32, len(tks)*m.embeddingDim)
	for i, tk := range tks {
		data, err := m.embeddingWeight.LoadBatch(tk)
		if err != nil {
			return nil, err
		}
		copy(ret[i*m.embeddingDim:], data)
	}
	return ret, nil
}

func (m *Model) Forward(embedding []float32, size int) ([]float32, error) {
	for layer := 0; layer < m.layers; layer++ {
		m.attention(layer, embedding, size)
	}
	return nil, nil
}

func (m *Model) attention(layer int, x []float32, size int) ([]float32, error) {
	inputShape := []int64{int64(size), int64(m.embeddingDim)}
	attentionShape := []int64{int64(m.embeddingDim), int64(m.embeddingDim)}
	attnQ := make([]float32, size*m.embeddingDim)
	attnK := make([]float32, size*m.embeddingDim)
	attnV := make([]float32, size*m.embeddingDim)
	wq, err := m.attentionWQ[layer].Load()
	if err != nil {
		return nil, fmt.Errorf("load layer%d.wq: %v", layer, err)
	}
	wk, err := m.attentionWK[layer].Load()
	if err != nil {
		return nil, fmt.Errorf("load layer%d.wk: %v", layer, err)
	}
	wv, err := m.attentionWV[layer].Load()
	if err != nil {
		return nil, fmt.Errorf("load layer%d.wv: %v", layer, err)
	}
	matMul(x, wq, inputShape, attentionShape, attnQ)
	matMul(x, wk, inputShape, attentionShape, attnK)
	matMul(x, wv, inputShape, attentionShape, attnV)
	// scale := 1 / math.Sqrt(float64(m.embeddingDim))
	return nil, nil
}
