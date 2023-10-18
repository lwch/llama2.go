package param

import (
	"archive/zip"
	"encoding/binary"
	"os"
	rt "runtime"
)

type Type byte

const (
	TypeBF16 Type = iota
	TypeQI8
	TypeQU8
	TypeQI7
	TypeQU7
	TypeQI6
	TypeQU6
	TypeQI5
	TypeQU5
	TypeQI4
	TypeQU4
)

func (t Type) Bits() int {
	switch t {
	case TypeQI8, TypeQU8:
		return 8
	case TypeQI7, TypeQU7:
		return 7
	case TypeQI6, TypeQU6:
		return 6
	case TypeQI5, TypeQU5:
		return 5
	case TypeQI4, TypeQU4:
		return 4
	default:
		return 16
	}
}

type Param interface {
	Shapes() []int64
	ElemCount() int64
	Dims() int64
	Warmup() error
	Load(cache bool) ([]float32, error)
	LoadBatch(uint64, []float32) error
}

type base struct {
	modelDir string
	fileName string
	shapes   []int64
}

func (b *base) init(modelDir, fileName string, shapes []int64) {
	b.modelDir = modelDir
	b.fileName = fileName
	b.shapes = shapes
}

func (b *base) Shapes() []int64 {
	return b.shapes
}

func (b *base) Dims() int64 {
	return int64(len(b.shapes))
}

func (b *base) ElemCount() int64 {
	n := b.shapes[0]
	for i := 1; i < len(b.shapes); i++ {
		n *= b.shapes[i]
	}
	return n
}

func (b *base) load(data any) error {
	f, err := os.Open(b.modelDir)
	if err != nil {
		return err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return err
	}
	zr, err := zip.NewReader(f, fi.Size())
	if err != nil {
		return err
	}
	fs, err := zr.Open(b.fileName)
	if err != nil {
		return err
	}
	defer fs.Close()
	switch dt := data.(type) {
	case []uint16:
		err = binary.Read(fs, binary.LittleEndian, dt)
	}
	rt.GC()
	if err != nil {
		return err
	}
	return nil
}
