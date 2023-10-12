package param

import (
	"unsafe"
)

type BF16 struct {
	base
	data []float32
}

var _ Param = &BF16{}

func NewBF16(modelDir, fileName string, shapes []int64) *BF16 {
	var ret BF16
	ret.init(modelDir, fileName, shapes)
	return &ret
}

func (bf16 *BF16) Load(cache bool) ([]float32, error) {
	if cache && bf16.data != nil {
		return bf16.data, nil
	}
	raw := make([]uint16, bf16.ElemCount())
	err := bf16.load(raw)
	if err != nil {
		return nil, err
	}
	ret := make([]float32, bf16.ElemCount())
	for i, v := range raw {
		ret[i] = decodeBFloat16(v)
	}
	if cache {
		bf16.data = ret
	}
	return ret, nil
}

func (bf16 *BF16) LoadBatch(n uint64) ([]float32, error) {
	batchSize := int64(1)
	if len(bf16.shapes) > 1 {
		batchSize = bf16.shapes[1]
		for i := 2; i < len(bf16.shapes); i++ {
			batchSize *= bf16.shapes[i]
		}
	}
	raw := make([]uint16, int64(n+1)*batchSize)
	err := bf16.load(raw)
	if err != nil {
		return nil, err
	}
	raw = raw[n*uint64(batchSize):]
	ret := make([]float32, batchSize)
	for i, v := range raw {
		ret[i] = decodeBFloat16(v)
	}
	return ret, nil
}

func encodeBFloat16(f float32) uint16 {
	n := *(*uint32)(unsafe.Pointer(&f))
	return uint16(n >> 16)
}

func decodeBFloat16(u16 uint16) float32 {
	n := uint32(u16) << 16
	return *(*float32)(unsafe.Pointer(&n))
}
