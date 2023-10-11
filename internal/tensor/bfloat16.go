package tensor

type BFloat16 struct {
	base
	data []uint16
}

var _ Tensor = &BFloat16{}

func NewBFloat16(shapes []int64, data []uint16) *BFloat16 {
	var ret BFloat16
	ret.shapes = shapes
	ret.data = data
	return &ret
}

func NewBFloat16Dup(shapes []int64, data []uint16) *BFloat16 {
	var ret BFloat16
	ret.shapes = dup(shapes)
	ret.data = dup(data)
	return &ret
}

func (t *BFloat16) MatMul(b Tensor) Tensor {
	// TODO: implement
	return nil
}

func (t *BFloat16) Raw() any {
	return t.data
}

func (t *BFloat16) BinaryData() any {
	return t.data
}
