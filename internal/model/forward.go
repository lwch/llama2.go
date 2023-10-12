package model

import (
	"fmt"
	"llama2/internal/math"
	math2 "math"
	"sync"
)

type Context struct {
	cacheK     [][]float32
	cacheV     [][]float32
	headSize   int64
	cacheParam bool
}

func (m *Model) NewContext(cacheParam bool) *Context {
	return &Context{
		cacheK:     make([][]float32, m.layers),
		cacheV:     make([][]float32, m.layers),
		headSize:   m.embeddingDim / m.heads,
		cacheParam: cacheParam,
	}
}

func (m *Model) Forward(ctx *Context, tk uint64, cursor int64) ([]float32, error) {
	x, err := m.embeddingWeight.LoadBatch(tk)
	if err != nil {
		return nil, err
	}
	for layer := 0; layer < m.layers; layer++ {
		x, err = m.attention(ctx, layer, x, cursor)
		if err != nil {
			return nil, err
		}
		x, err = m.feedForward(ctx, x, layer)
		if err != nil {
			return nil, err
		}
	}

	norm, err := m.norm.Load(ctx.cacheParam) // (dim)
	if err != nil {
		return nil, fmt.Errorf("load norm: %v", err)
	}
	output, err := m.output.Load(ctx.cacheParam) // (dim, vocab_size)
	if err != nil {
		return nil, fmt.Errorf("load output: %v", err)
	}

	// rmsnorm
	y := make([]float32, m.embeddingDim) // (1, dim)
	math.RMSNorm(x, norm, y, m.eps)      // (1, dim)

	vocabSize := m.output.Shapes()[1]

	// y @ output
	// (1, dim) @ (dim, vocab_size) => (1, vocab_size)
	z := make([]float32, vocabSize)                         // (1, vocab_size)
	math.MatMul(y, output, 1, vocabSize, m.embeddingDim, z) // (1, vocab_size)

	return z, nil
}

func (m *Model) attention(ctx *Context, layer int, x []float32, cursor int64) ([]float32, error) {
	seqlen := cursor + 1
	attnQ := make([]float32, m.embeddingDim)
	attnK := make([]float32, m.embeddingDim)
	attnV := make([]float32, m.embeddingDim)
	wq, err := m.attentionWQ[layer].Load(ctx.cacheParam) // (dim, dim)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.attention_wq: %v", layer, err)
	}
	wk, err := m.attentionWK[layer].Load(ctx.cacheParam) // (dim, dim)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.attention_wk: %v", layer, err)
	}
	wv, err := m.attentionWV[layer].Load(ctx.cacheParam) // (dim, dim)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.attention_wv: %v", layer, err)
	}
	wo, err := m.attentionWO[layer].Load(ctx.cacheParam) // (dim, dim)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.attention_wo: %v", layer, err)
	}
	norm, err := m.attentionNorm[layer].Load(ctx.cacheParam) // (dim)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.attention_norm: %v", layer, err)
	}

	dx := make([]float32, m.embeddingDim) // (1, dim)
	math.RMSNorm(x, norm, dx, m.eps)      // (1, dim)

	// compute q, k, v vector
	math.MatMul(dx, wq, 1, m.embeddingDim, m.embeddingDim, attnQ) // (1, dim)
	math.MatMul(dx, wk, 1, m.embeddingDim, m.embeddingDim, attnK) // (1, dim)
	math.MatMul(dx, wv, 1, m.embeddingDim, m.embeddingDim, attnV) // (1, dim)
	clear(dx)

	math.ROPE(attnQ, attnK, cursor, ctx.headSize)

	// append cache
	attnK = append(ctx.cacheK[layer], attnK...) // (seqlen, dim)
	attnV = append(ctx.cacheV[layer], attnV...) // (seqlen, dim)
	ctx.cacheK[layer] = attnK
	ctx.cacheV[layer] = attnV

	// attention
	scale := math2.Sqrt(float64(ctx.headSize))
	var wgHeads sync.WaitGroup
	wgHeads.Add(int(m.heads))
	for head := int64(0); head < m.heads; head++ {
		go func(head int64) {
			defer wgHeads.Done()

			// q @ k^T
			// (1, head_size) @ (head_size, seqlen) => (1, seqlen)
			q := attnQ[head*ctx.headSize : (head+1)*ctx.headSize] // (1, head_size)
			score := make([]float32, seqlen)                      // (seqlen)
			var wg sync.WaitGroup
			wg.Add(int(seqlen))
			for cursor := int64(0); cursor < seqlen; cursor++ {
				go func(cursor int64) {
					defer wg.Done()
					idx := cursor * m.embeddingDim
					k := attnK[idx+head*ctx.headSize : idx+(head+1)*ctx.headSize] // (head_size, 1)
					math.MatMul(q, k, 1, 1, ctx.headSize, score[cursor:cursor+1]) // (1, head_size) @ (head_size, 1) => (1)
					score[cursor] /= float32(scale)
				}(cursor)
			}
			wg.Wait()

			// softmax
			// (seqlen)
			math.Softmax(score, seqlen)

			// score @ v
			// (seqlen) @ (seqlen, head_size) => (head_size)
			wg.Add(int(seqlen * ctx.headSize))
			for cursor := int64(0); cursor < seqlen; cursor++ {
				idx := cursor * m.embeddingDim
				for dim := int64(0); dim < ctx.headSize; dim++ {
					go func(cursor, dim int64) {
						defer wg.Done()
						dx[head*ctx.headSize+dim] += score[cursor] * attnV[idx+head*ctx.headSize+dim]
					}(cursor, dim)
				}
			}
			wg.Wait()
		}(head)
	}
	wgHeads.Wait()

	// dx @ wo
	// (1, dim) @ (dim, dim) => (1, dim)
	y := make([]float32, m.embeddingDim)                      // (1, dim)
	math.MatMul(dx, wo, 1, m.embeddingDim, m.embeddingDim, y) // (1, dim)

	// residual connection
	for i := range x {
		y[i] += x[i]
	}
	return y, nil
}

func (m *Model) feedForward(ctx *Context, x []float32, layer int) ([]float32, error) {
	w1, err := m.ffnW1[layer].Load(ctx.cacheParam) // (dim, dim2)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.feed_forward_w1: %v", layer, err)
	}
	w2, err := m.ffnW2[layer].Load(ctx.cacheParam) // (dim2, dim)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.feed_forward_w2: %v", layer, err)
	}
	w3, err := m.ffnW3[layer].Load(ctx.cacheParam) // (dim, dim2)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.feed_forward_w3: %v", layer, err)
	}
	norm, err := m.ffnNorm[layer].Load(ctx.cacheParam) // (dim)
	if err != nil {
		return nil, fmt.Errorf("load layer%d.feed_forward_norm: %v", layer, err)
	}

	if m.ffnW1[layer].Shapes()[1] != m.ffnW3[layer].Shapes()[1] ||
		m.ffnW1[layer].Shapes()[1] != m.ffnW2[layer].Shapes()[0] {
		return nil, fmt.Errorf("invalid feed forward weight shape")
	}
	dim2 := m.ffnW1[layer].Shapes()[1]

	dx := make([]float32, m.embeddingDim) // (1, dim)
	math.RMSNorm(x, norm, dx, m.eps)      // (1, dim)

	// dx @ w1
	// (1, dim) @ (dim, dim2) => (1, dim2)
	y1 := make([]float32, dim2)                      // (1, dim2)
	math.MatMul(dx, w1, 1, dim2, m.embeddingDim, y1) // (1, dim2)

	// dx @ w3
	// (1, dim) @ (dim, dim2) => (1, dim2)
	y2 := make([]float32, dim2)                      // (1, dim2)
	math.MatMul(dx, w3, 1, dim2, m.embeddingDim, y2) // (1, dim2)

	// silu for y1
	// (1, dim2)
	math.SiLU(y1) // (1, dim2)

	// y1 * y2
	// (1, dim2) * (1, dim2) => (1, dim2)
	y := make([]float32, dim2) // (1, dim2)
	math.Mul(y1, y2, y)        // (1, dim2)

	// y @ w2
	// (1, dim2) @ (dim2, dim) => (1, dim)
	clear(dx)
	math.MatMul(y, w2, 1, m.embeddingDim, dim2, dx) // (1, dim)

	// residual connection
	for i := range x {
		dx[i] += x[i]
	}
	return dx, nil
}

func clear(x []float32) {
	for i := range x {
		x[i] = 0
	}
}
