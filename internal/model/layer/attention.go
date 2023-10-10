package layer

import (
	"llama2/internal/model/parallel"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/net"
)

type Attention struct {
	Module
	wq    *tensor.Tensor
	wk    *tensor.Tensor
	wv    *tensor.Tensor
	wo    *tensor.Tensor
	heads int64
	dims  int64
}

var _ layer.Layer = &Attention{}

func NewAttention(wq, wk, wv, wo *tensor.Tensor, heads, dims int64) *Attention {
	return &Attention{
		wq:    wq,
		wk:    wk,
		wv:    wv,
		wo:    wo,
		heads: heads,
		dims:  dims,
	}
}

func reshapeForBroadcast(freqs, x *tensor.Tensor) *tensor.Tensor {
	shapes := x.Shapes()
	for i := 0; i < len(shapes)-1; i++ {
		if i == 1 || i == len(shapes)-1 {
			continue
		} else {
			shapes[i] = 1
		}
	}
	return freqs.View(shapes...)
}

func applyRotaryEmb(xq, xk, freqs *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	shapes := xq.Shapes()
	shapes = shapes[:len(shapes)-1]
	shapes = append(shapes, -1, 2)
	xq_ := xq.ToScalarType(consts.KHalf).Reshape(shapes...).ViewAsComplex()

	shapes = xk.Shapes()
	shapes = shapes[:len(shapes)-1]
	shapes = append(shapes, -1, 2)
	xk_ := xk.ToScalarType(consts.KHalf).Reshape(shapes...).ViewAsComplex()

	freqs = reshapeForBroadcast(freqs, xq_)
	xqOut := xq_.Mul(freqs).ViewAsReal().View(xq.Shapes()...)
	xkOut := xk_.Mul(freqs).ViewAsReal().View(xk.Shapes()...)
	return xqOut.ToScalarType(xq.ScalarType()), xkOut.ToScalarType(xk.ScalarType())
}

func (l *Attention) Forward(x, freqs *tensor.Tensor) *tensor.Tensor {
	bsz, seqlen := x.Shapes()[0], x.Shapes()[1]
	xq, xk, xv := parallel.MatMul(x, l.wq),
		parallel.MatMul(x, l.wk),
		parallel.MatMul(x, l.wv)

	headDim := l.dims / l.heads
	xq = xq.View(bsz, seqlen, l.heads, headDim)
	xk = xk.View(bsz, seqlen, l.heads, headDim)
	xv = xv.View(bsz, seqlen, l.heads, headDim)

	xq, xk = applyRotaryEmb(xq, xk, freqs)

	xq = xq.Transpose(1, 2) // (bs, heads, seqlen, head_dim)
	xk = xk.Transpose(1, 2) // (bs, heads, seqlen, head_dim)
	xv = xv.Transpose(1, 2) // (bs, heads, seqlen, head_dim)

	output := tensor.ScaledDotProductAttention(xq, xk, xv, nil, 0, true) // (bs, heads, seqlen, head_dim)
	output = output.Transpose(1, 2).Contiguous().View(bsz, seqlen, -1)

	return parallel.MatMul(output, l.wo)
}

func init() {
	net.RegisterLoadFunc("llama2.attention", func(name string, params map[string]*tensor.Tensor, args map[string]float32) layer.Layer {
		var layer Attention
		layer.wq = params["wq"]
		layer.wk = params["wk"]
		layer.wv = params["wv"]
		layer.wo = params["wo"]
		layer.heads = int64(args["heads"])
		layer.dims = int64(args["dims"])
		return &layer
	})
}

func (l *Attention) Class() string {
	return "llama2.attention"
}

func (l *Attention) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"wq": l.wq,
		"wk": l.wk,
		"wv": l.wv,
		"wo": l.wo,
	}
}

func (l *Attention) Args() map[string]float32 {
	return map[string]float32{
		"heads": float32(l.heads),
		"dims":  float32(l.dims),
	}
}

func (l *Attention) Freeze() {
	l.wq.SetRequiresGrad(false)
	l.wk.SetRequiresGrad(false)
	l.wv.SetRequiresGrad(false)
	l.wo.SetRequiresGrad(false)
}

func (l *Attention) Unfreeze() {
	l.wq.SetRequiresGrad(true)
	l.wk.SetRequiresGrad(true)
	l.wv.SetRequiresGrad(true)
	l.wo.SetRequiresGrad(true)
}

func (l *Attention) ToScalarType(t consts.ScalarType) *Attention {
	var layer Attention
	layer.wq = l.wq.ToScalarType(t)
	layer.wk = l.wk.ToScalarType(t)
	layer.wv = l.wv.ToScalarType(t)
	layer.wo = l.wo.ToScalarType(t)
	layer.heads = l.heads
	layer.dims = l.dims
	return &layer
}
