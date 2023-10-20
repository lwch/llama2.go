package model

import (
	"fmt"
	"llama2/internal/math"
	"sync"

	"github.com/chewxy/math32"
)

const maxInferenceLength = 2048 // max inference length of LLAMA2

type Context struct {
	cacheK     [][]float32
	cacheV     [][]float32
	headDim    int64
	qDim       int64
	kvDim      int64
	cacheParam bool
	fp32       bool
	// global
	x  []float32
	dx []float32
	y  []float32
	// attention
	attnQ, attnK, attnV []float32
	scores              [][maxInferenceLength]float32
	attnY               []float32
	// ffn
	ffnLeft, ffnRight []float32
	ffnY              []float32
}

func (m *Model) NewContext(cacheParam, fp32 bool) *Context {
	headDim := m.embeddingDim / m.heads
	qDim := m.heads * headDim
	kvDim := m.kvHeads * headDim
	dim2 := m.ffnW1[0].Shapes()[0]
	cacheK := make([][]float32, m.layers)
	cacheV := make([][]float32, m.layers)
	for i := range cacheK {
		cacheK[i] = make([]float32, maxInferenceLength*kvDim)
		cacheV[i] = make([]float32, maxInferenceLength*kvDim)
	}
	return &Context{
		cacheK:     cacheK,
		cacheV:     cacheV,
		headDim:    headDim,
		qDim:       qDim,
		kvDim:      kvDim,
		cacheParam: cacheParam,
		// global
		x:  make([]float32, m.embeddingDim),
		dx: make([]float32, m.embeddingDim),
		y:  make([]float32, m.vocabSize),
		// attention
		attnQ:  make([]float32, qDim),
		attnK:  make([]float32, kvDim),
		attnV:  make([]float32, kvDim),
		scores: make([][maxInferenceLength]float32, m.heads),
		attnY:  make([]float32, m.embeddingDim),
		// ffn
		ffnLeft:  make([]float32, dim2),
		ffnRight: make([]float32, dim2),
		ffnY:     make([]float32, m.embeddingDim),
	}
}

func (m *Model) Forward(ctx *Context, tk uint64, cursor int64) ([]float32, error) {
	err := m.embeddingWeight.LoadBatch(tk, ctx.x)
	if err != nil {
		return nil, err
	}
	for layer := int64(0); layer < m.layers; layer++ {
		err = m.attention(ctx, layer, ctx.x, cursor)
		if err != nil {
			return nil, err
		}
		err = m.feedForward(ctx, ctx.x, layer)
		if err != nil {
			return nil, err
		}
	}

	norm, err := m.norm.Load(ctx.cacheParam, ctx.fp32) // (dim)
	if err != nil {
		return nil, fmt.Errorf("load norm: %v", err)
	}
	output, err := m.output.Load(ctx.cacheParam, ctx.fp32) // (dim, vocab_size)
	if err != nil {
		return nil, fmt.Errorf("load output: %v", err)
	}

	// rmsnorm
	math.RMSNorm(ctx.x, norm, ctx.dx, m.eps) // (1, dim)

	// y @ output
	// (1, dim) @ (dim, vocab_size) => (1, vocab_size)
	math.MatMul(ctx.dx, output, 1, m.vocabSize, m.embeddingDim, ctx.y) // (1, vocab_size)

	return ctx.y, nil
}

func (m *Model) attention(ctx *Context, layer int64, x []float32, cursor int64) error {
	seqlen := cursor + 1
	wq, err := m.attentionWQ[layer].Load(ctx.cacheParam, ctx.fp32) // (dim, q_dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_wq: %v", layer, err)
	}
	wk, err := m.attentionWK[layer].Load(ctx.cacheParam, ctx.fp32) // (dim, kv_dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_wk: %v", layer, err)
	}
	wv, err := m.attentionWV[layer].Load(ctx.cacheParam, ctx.fp32) // (dim, kv_dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_wv: %v", layer, err)
	}
	wo, err := m.attentionWO[layer].Load(ctx.cacheParam, ctx.fp32) // (dim, dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_wo: %v", layer, err)
	}
	norm, err := m.attentionNorm[layer].Load(ctx.cacheParam, ctx.fp32) // (dim)
	if err != nil {
		return fmt.Errorf("load layer%d.attention_norm: %v", layer, err)
	}

	math.RMSNorm(x, norm, ctx.dx, m.eps) // (1, dim)

	// compute q, k, v vector
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		math.MatMul(ctx.dx, wq, 1, ctx.qDim, m.embeddingDim, ctx.attnQ) // (1, q_dim)
	}()
	go func() {
		defer wg.Done()
		math.MatMul(ctx.dx, wk, 1, ctx.kvDim, m.embeddingDim, ctx.attnK) // (1, kv_dim)
	}()
	go func() {
		defer wg.Done()
		math.MatMul(ctx.dx, wv, 1, ctx.kvDim, m.embeddingDim, ctx.attnV) // (1, kv_dim)
	}()
	wg.Wait()
	math.Clear(ctx.dx)

	math.ROPE(ctx.attnQ, ctx.attnK, cursor, ctx.headDim)

	// append cache
	idx := cursor * ctx.kvDim
	copy(ctx.cacheK[layer][idx:idx+ctx.kvDim], ctx.attnK)
	copy(ctx.cacheV[layer][idx:idx+ctx.kvDim], ctx.attnV)
	attnK := ctx.cacheK[layer][:seqlen*ctx.kvDim] // (seqlen, kv_dim)
	attnV := ctx.cacheV[layer][:seqlen*ctx.kvDim] // (seqlen, kv_dim)

	// attention
	scale := math32.Sqrt(float32(ctx.headDim))
	for head := int64(0); head < m.heads; head++ {
		headIdx := head * ctx.headDim
		offsetKV := (head % m.kvHeads) * ctx.headDim

		// q @ k^T
		// (1, head_dim) @ (head_dim, seqlen) => (1, seqlen)
		q := ctx.attnQ[headIdx : headIdx+ctx.headDim] // (1, head_dim)
		score := ctx.scores[head]                     // (seqlen)
		for cursor := int64(0); cursor < seqlen; cursor++ {
			idx := cursor*m.kvHeads*ctx.headDim + offsetKV
			k := attnK[idx : idx+ctx.headDim]                            // (head_dim, 1), repeat k if kv_heads < heads
			math.MatMul(q, k, 1, 1, ctx.headDim, score[cursor:cursor+1]) // (1, head_size) @ (head_size, 1) => (1)
			score[cursor] /= scale
		}

		// softmax
		// (seqlen)
		math.Softmax(score[:], seqlen)

		offsetX := head * ctx.headDim
		dx := ctx.dx[offsetX : offsetX+ctx.headDim]
		idx = offsetKV
		// score @ v
		// (seqlen) @ (seqlen, head_dim) => (head_dim)
		for cursor := int64(0); cursor < seqlen; cursor++ {
			math.Axpy(score[cursor], attnV[idx:idx+ctx.headDim], dx)
			idx += m.kvHeads * ctx.headDim
		}
	}

	// dx @ wo
	// (1, dim) @ (dim, dim) => (1, dim)
	math.MatMul(ctx.dx, wo, 1, m.embeddingDim, m.embeddingDim, ctx.attnY) // (1, dim)

	// residual connection
	math.Add(x, ctx.attnY) // (1, dim)
	return nil
}

func (m *Model) feedForward(ctx *Context, x []float32, layer int64) error {
	w1, err := m.ffnW1[layer].Load(ctx.cacheParam, ctx.fp32) // (dim, dim2)
	if err != nil {
		return fmt.Errorf("load layer%d.feed_forward_w1: %v", layer, err)
	}
	w2, err := m.ffnW2[layer].Load(ctx.cacheParam, ctx.fp32) // (dim2, dim)
	if err != nil {
		return fmt.Errorf("load layer%d.feed_forward_w2: %v", layer, err)
	}
	w3, err := m.ffnW3[layer].Load(ctx.cacheParam, ctx.fp32) // (dim, dim2)
	if err != nil {
		return fmt.Errorf("load layer%d.feed_forward_w3: %v", layer, err)
	}
	norm, err := m.ffnNorm[layer].Load(ctx.cacheParam, ctx.fp32) // (dim)
	if err != nil {
		return fmt.Errorf("load layer%d.feed_forward_norm: %v", layer, err)
	}

	if m.ffnW1[layer].Shapes()[0] != m.ffnW3[layer].Shapes()[0] ||
		m.ffnW1[layer].Shapes()[0] != m.ffnW2[layer].Shapes()[1] {
		return fmt.Errorf("invalid feed forward weight shape")
	}
	dim2 := m.ffnW1[layer].Shapes()[0]

	math.RMSNorm(x, norm, ctx.dx, m.eps) // (1, dim)

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()

		// dx @ w1
		// (1, dim) @ (dim, dim2) => (1, dim2)
		math.MatMul(ctx.dx, w1, 1, dim2, m.embeddingDim, ctx.ffnLeft) // (1, dim2)

		// silu for y1
		// (1, dim2)
		math.SiLU(ctx.ffnLeft) // (1, dim2)
	}()

	go func() {
		defer wg.Done()

		// dx @ w3
		// (1, dim) @ (dim, dim2) => (1, dim2)
		math.MatMul(ctx.dx, w3, 1, dim2, m.embeddingDim, ctx.ffnRight) // (1, dim2)
	}()

	wg.Wait()

	// y1 * y2
	// (1, dim2) * (1, dim2) => (1, dim2)
	math.Mul(ctx.ffnLeft, ctx.ffnRight) // (1, dim2)

	// y @ w2
	// (1, dim2) @ (dim2, dim) => (1, dim)
	math.MatMul(ctx.ffnLeft, w2, 1, m.embeddingDim, dim2, ctx.ffnY) // (1, dim)

	// residual connection
	math.Add(x, ctx.ffnY) // (1, dim)
	return nil
}
