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
	headDim    int64
	qDim       int64
	kvDim      int64
	cacheParam bool
	// global
	x  []float32
	dx []float32
	y  []float32
	// attention
	attnQ, attnK, attnV []float32
	score               []float32
	yAttn               []float32
	// ffn
	left, right []float32
}

func (m *Model) NewContext(cacheParam bool) *Context {
	headDim := m.embeddingDim / m.heads
	qDim := m.heads * headDim
	kvDim := m.kvHeads * headDim
	dim2 := m.ffnW1[0].Shapes()[1]
	return &Context{
		cacheK:     make([][]float32, m.layers),
		cacheV:     make([][]float32, m.layers),
		headDim:    headDim,
		qDim:       qDim,
		kvDim:      kvDim,
		cacheParam: cacheParam,
		// global
		x:  make([]float32, m.embeddingDim),
		dx: make([]float32, m.embeddingDim),
		y:  make([]float32, m.vocabSize),
		// attention
		attnQ: make([]float32, qDim),
		attnK: make([]float32, kvDim),
		attnV: make([]float32, kvDim),
		score: make([]float32, 2048), // max of llama sequence length
		yAttn: make([]float32, m.embeddingDim),
		// ffn
		left:  make([]float32, dim2),
		right: make([]float32, dim2),
	}
}

func (m *Model) Forward(ctx *Context, tk uint64, cursor int64) ([]float32, error) {
	err := m.embeddingWeight.LoadBatch(tk, ctx.x)
	if err != nil {
		return nil, err
	}
	for layer := 0; layer < m.layers; layer++ {
		err = m.attention(ctx, layer, ctx.x, cursor)
		if err != nil {
			return nil, err
		}
		err = m.feedForward(ctx, ctx.x, layer)
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
	math.RMSNorm(ctx.x, norm, ctx.dx, m.eps) // (1, dim)

	vocabSize := m.output.Shapes()[1]

	// y @ output
	// (1, dim) @ (dim, vocab_size) => (1, vocab_size)
	math.MatMul(ctx.dx, output, 1, vocabSize, m.embeddingDim, ctx.y) // (1, vocab_size)

	return ctx.y, nil
}

func (m *Model) attention(ctx *Context, layer int, x []float32, cursor int64) error {
	seqlen := cursor + 1
	wq, err := m.attentionWQ[layer].Load(ctx.cacheParam) // (dim, q_dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_wq: %v", layer, err)
	}
	wk, err := m.attentionWK[layer].Load(ctx.cacheParam) // (dim, kv_dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_wk: %v", layer, err)
	}
	wv, err := m.attentionWV[layer].Load(ctx.cacheParam) // (dim, kv_dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_wv: %v", layer, err)
	}
	wo, err := m.attentionWO[layer].Load(ctx.cacheParam) // (dim, dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_wo: %v", layer, err)
	}
	norm, err := m.attentionNorm[layer].Load(ctx.cacheParam) // (dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_norm: %v", layer, err)
	}

	math.RMSNorm(x, norm, ctx.dx, m.eps) // (1, dim)

	// compute q, k, v vector
	math.MatMul(ctx.dx, wq, 1, ctx.qDim, m.embeddingDim, ctx.attnQ)  // (1, q_dim)
	math.MatMul(ctx.dx, wk, 1, ctx.kvDim, m.embeddingDim, ctx.attnK) // (1, kv_dim)
	math.MatMul(ctx.dx, wv, 1, ctx.kvDim, m.embeddingDim, ctx.attnV) // (1, kv_dim)
	clear(ctx.dx)

	math.ROPE(ctx.attnQ, ctx.attnK, cursor, ctx.headDim)

	// append cache
	attnK := append(ctx.cacheK[layer], ctx.attnK...) // (seqlen, kv_dim)
	attnV := append(ctx.cacheV[layer], ctx.attnV...) // (seqlen, kv_dim)
	ctx.cacheK[layer] = attnK
	ctx.cacheV[layer] = attnV

	// attention
	scale := math2.Sqrt(float64(ctx.headDim))
	var wgHeads sync.WaitGroup
	wgHeads.Add(int(m.heads))
	for head := int64(0); head < m.heads; head++ {
		go func(head int64) {
			defer wgHeads.Done()

			// q @ k^T
			// (1, head_dim) @ (head_dim, seqlen) => (1, seqlen)
			q := ctx.attnQ[head*ctx.headDim : (head+1)*ctx.headDim] // (1, head_dim)
			score := ctx.score[:seqlen]                             // (seqlen)
			var wg sync.WaitGroup
			wg.Add(int(seqlen))
			for cursor := int64(0); cursor < seqlen; cursor++ {
				go func(cursor int64) {
					defer wg.Done()
					idx := cursor * m.kvHeads * ctx.headDim
					k := attnK[idx+(head%m.kvHeads)*ctx.headDim : idx+((head%m.kvHeads)+1)*ctx.headDim] // (head_dim, 1), repeat k if kv_heads < heads
					math.MatMul(q, k, 1, 1, ctx.headDim, score[cursor:cursor+1])                        // (1, head_size) @ (head_size, 1) => (1)
					score[cursor] /= float32(scale)
				}(cursor)
			}
			wg.Wait()

			// softmax
			// (seqlen)
			math.Softmax(score, seqlen)

			// score @ v
			// (seqlen) @ (seqlen, head_dim) => (head_dim)
			wg.Add(int(seqlen * ctx.headDim))
			for cursor := int64(0); cursor < seqlen; cursor++ {
				idx := cursor * m.kvHeads * ctx.headDim
				for dim := int64(0); dim < ctx.headDim; dim++ {
					go func(cursor, dim int64) {
						defer wg.Done()
						ctx.dx[head*ctx.headDim+dim] += score[cursor] * attnV[idx+(head%m.kvHeads)*ctx.headDim+dim] // repeat v if kv_heads < heads
					}(cursor, dim)
				}
			}
			wg.Wait()
		}(head)
	}
	wgHeads.Wait()

	// dx @ wo
	// (1, dim) @ (dim, dim) => (1, dim)
	math.MatMul(ctx.dx, wo, 1, m.embeddingDim, m.embeddingDim, ctx.yAttn) // (1, dim)

	// residual connection
	for i := range x {
		x[i] += ctx.yAttn[i]
	}
	return nil
}

func (m *Model) feedForward(ctx *Context, x []float32, layer int) error {
	w1, err := m.ffnW1[layer].Load(ctx.cacheParam) // (dim, dim2)
	if err != nil {
		return fmt.Errorf("load layer%d.feed_forward_w1: %v", layer, err)
	}
	w2, err := m.ffnW2[layer].Load(ctx.cacheParam) // (dim2, dim)
	if err != nil {
		return fmt.Errorf("load layer%d.feed_forward_w2: %v", layer, err)
	}
	w3, err := m.ffnW3[layer].Load(ctx.cacheParam) // (dim, dim2)
	if err != nil {
		return fmt.Errorf("load layer%d.feed_forward_w3: %v", layer, err)
	}
	norm, err := m.ffnNorm[layer].Load(ctx.cacheParam) // (dim)
	if err != nil {
		return fmt.Errorf("load layer%d.feed_forward_norm: %v", layer, err)
	}

	if m.ffnW1[layer].Shapes()[1] != m.ffnW3[layer].Shapes()[1] ||
		m.ffnW1[layer].Shapes()[1] != m.ffnW2[layer].Shapes()[0] {
		return fmt.Errorf("invalid feed forward weight shape")
	}
	dim2 := m.ffnW1[layer].Shapes()[1]

	math.RMSNorm(x, norm, ctx.dx, m.eps) // (1, dim)

	// dx @ w1
	// (1, dim) @ (dim, dim2) => (1, dim2)
	math.MatMul(ctx.dx, w1, 1, dim2, m.embeddingDim, ctx.left) // (1, dim2)

	// dx @ w3
	// (1, dim) @ (dim, dim2) => (1, dim2)
	math.MatMul(ctx.dx, w3, 1, dim2, m.embeddingDim, ctx.right) // (1, dim2)

	// silu for y1
	// (1, dim2)
	math.SiLU(ctx.left) // (1, dim2)

	// y1 * y2
	// (1, dim2) * (1, dim2) => (1, dim2)
	math.Mul(ctx.left, ctx.right) // (1, dim2)

	// y @ w2
	// (1, dim2) @ (dim2, dim) => (1, dim)
	math.MatMul(ctx.left, w2, 1, m.embeddingDim, dim2, ctx.dx) // (1, dim)

	// residual connection
	for i := range x {
		x[i] += ctx.dx[i]
	}
	return nil
}

func clear(x []float32) {
	for i := range x {
		x[i] = 0
	}
}
