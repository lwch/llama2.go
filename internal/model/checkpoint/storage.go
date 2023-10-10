package checkpoint

import (
	"archive/zip"
	"sync"
)

type storageType byte

const (
	typeBFloat16 storageType = iota // torch.bfloat16
	typeHalf                        // torch.half, torch.float16
	typeFloat                       // torch.float, torch.float32
	typeDouble                      // torch.double, torch.float64
	typeByte                        // torch.byte, torch.uint8
	typeChar                        // torch.char, torch.int8
	typeShort                       // torch.short, torch.int16
	typeInt                         // torch.int, torch.int32
	typeLong                        // torch.long, torch.int64
)

type storage interface {
	New(wg *sync.WaitGroup, size int, file *zip.File) (storage, error)
	SetShape(shape []int64)
	GetShape() []int64
	SetRequiresGrad(requiresGrad bool)
	GetRequiresGrad() bool
	Type() storageType
	Get() interface{}
}

type base struct {
	shape        []int64
	requiresGrad bool
}

func (b *base) SetShape(shape []int64) {
	b.shape = shape
}

func (b *base) GetShape() []int64 {
	return b.shape
}

func (b *base) SetRequiresGrad(requiresGrad bool) {
	b.requiresGrad = requiresGrad
}

func (b *base) GetRequiresGrad() bool {
	return b.requiresGrad
}
