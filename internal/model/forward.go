package model

import (
	"fmt"
	"sync"
)

func (m *Model) LookupEmbedding(tks []uint64) ([]float32, error) {
	ret := make([]float32, len(tks)*int(m.embeddingDim))
	var wg sync.WaitGroup
	wg.Add(len(tks))
	var err error
	for i, tk := range tks {
		go func(i int, tk uint64) {
			defer wg.Done()
			data, e := m.embeddingWeight.LoadBatch(tk)
			if err != nil {
				err = e
			}
			copy(ret[i*int(m.embeddingDim):], data)
		}(i, tk)
	}
	wg.Wait()
	return ret, nil
}

func (m *Model) Forward(x []float32, seqlen int64) ([]float32, error) {
	for layer := 0; layer < m.layers; layer++ {
		m.attention(layer, x, seqlen)
	}
	return nil, nil
}

func (m *Model) attention(layer int, x []float32, seqlen int64) ([]float32, error) {
	attnQ := make([]float32, seqlen*m.embeddingDim)
	attnK := make([]float32, seqlen*m.embeddingDim)
	attnV := make([]float32, seqlen*m.embeddingDim)
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
	matMul(x, wq, seqlen, m.embeddingDim, m.embeddingDim, attnQ)
	matMul(x, wk, seqlen, m.embeddingDim, m.embeddingDim, attnK)
	matMul(x, wv, seqlen, m.embeddingDim, m.embeddingDim, attnV)
	// scale := 1 / math.Sqrt(float64(m.embeddingDim))
	return nil, nil
}
