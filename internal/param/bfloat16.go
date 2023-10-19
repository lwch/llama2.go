package param

import (
	"llama2/internal/utils"
	"runtime"
	"unsafe"
)

type BF16 struct {
	base
	dataHalf []uint16
	dataFP32 []float32
}

var _ Param = &BF16{}

func NewBF16(modelDir, fileName string, shapes []int64) *BF16 {
	var ret BF16
	ret.init(modelDir, fileName, shapes)
	return &ret
}

func (bf16 *BF16) decode(data []uint16) []float32 {
	ret := make([]float32, len(data))
	utils.Parallel(len(data), runtime.NumCPU(), func(_, offset, size int) {
		for i := 0; i < size; i++ {
			decodeBFloat16(&ret[offset+i], data[offset+i])
		}
	})
	return ret
}

func (bf16 *BF16) Warmup(fp32 bool) error {
	if bf16.dataHalf != nil {
		return nil
	}
	if bf16.dataFP32 != nil {
		return nil
	}
	data := make([]uint16, bf16.ElemCount())
	err := bf16.load(data)
	if err != nil {
		return err
	}
	if fp32 {
		bf16.dataFP32 = bf16.decode(data)
		return nil
	}
	bf16.dataHalf = data
	return nil
}

func (bf16 *BF16) Load(cache, fp32 bool) ([]float32, error) {
	if bf16.dataFP32 != nil {
		return bf16.dataFP32, nil
	}
	if bf16.dataHalf != nil {
		return bf16.decode(bf16.dataHalf), nil
	}
	raw := make([]uint16, bf16.ElemCount())
	err := bf16.load(raw)
	if err != nil {
		return nil, err
	}
	if cache {
		if fp32 {
			bf16.dataFP32 = bf16.decode(raw)
			return bf16.dataFP32, nil
		}
		bf16.dataHalf = raw
		return bf16.decode(raw), nil
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
		decodeBFloat16(&data[i], v)
	}
	return nil
}

func encodeBFloat16(f float32) uint16 {
	n := *(*uint32)(unsafe.Pointer(&f))
	return uint16(n >> 16)
}

func decodeBFloat16(p *float32, u16 uint16) {
	u32 := (*uint32)(unsafe.Pointer(p))
	*u32 = uint32(u16) << 16
}
