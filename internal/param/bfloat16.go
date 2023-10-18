package param

import (
	"unsafe"
)

type BF16 struct {
	base
	data []uint16
}

var _ Param = &BF16{}

func NewBF16(modelDir, fileName string, shapes []int64) *BF16 {
	var ret BF16
	ret.init(modelDir, fileName, shapes)
	return &ret
}

func (bf16 *BF16) decode(data []uint16) []float32 {
	ret := make([]float32, len(data))
	for i := 0; i < len(data); {
		ret[i] = decodeBFloat16(data[i])
	}
	return ret
}

func (bf16 *BF16) Warmup() error {
	if bf16.data != nil {
		return nil
	}
	bf16.data = make([]uint16, bf16.ElemCount())
	err := bf16.load(bf16.data)
	if err != nil {
		return err
	}
	return nil
}

func (bf16 *BF16) Load(cache bool) ([]float32, error) {
	if cache && bf16.data != nil {
		return bf16.decode(bf16.data), nil
	}
	raw := make([]uint16, bf16.ElemCount())
	err := bf16.load(raw)
	if err != nil {
		return nil, err
	}
	if cache {
		bf16.data = raw
	}
	return bf16.decode(raw), nil
}

func (bf16 *BF16) LoadBatch(n uint64, data []float32) error {
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
		return err
	}
	raw = raw[n*uint64(batchSize):]
	for i, v := range raw {
		data[i] = decodeBFloat16(v)
	}
	return nil
}

func encodeBFloat16(f float32) uint16 {
	n := *(*uint32)(unsafe.Pointer(&f))
	return uint16(n >> 16)
}

func decodeBFloat16(u16 uint16) float32 {
	n := uint32(u16) << 16
	return *(*float32)(unsafe.Pointer(&n))
}
