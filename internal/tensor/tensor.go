package tensor

type Type byte

const (
	TypeBF16 Type = iota
)

type Tensor interface {
	Shapes() []int64
	Dims() int
	Raw() any
	BinaryData() any
	MatMul(b Tensor) Tensor
}

type base struct {
	shapes []int64
}

func (b *base) Shapes() []int64 {
	return b.shapes
}

func (b *base) Dims() int {
	return len(b.shapes)
}
