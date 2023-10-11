package checkpoint

import (
	"archive/zip"
	"encoding/binary"
	"os"
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

type Storage interface {
	New(ckptDir, fileName string, dataSize int) Storage
	SetShape(shape []int64)
	GetShape() []int64
	SetRequiresGrad(requiresGrad bool)
	GetRequiresGrad() bool
	Type() storageType
	Load() (any, error)
}

type base struct {
	ckptDir      string
	fileName     string
	dataSize     int
	shape        []int64
	requiresGrad bool
}

func (b *base) init(ckptDir, fileName string, dataSize int) {
	b.ckptDir = ckptDir
	b.fileName = fileName
	b.dataSize = dataSize
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

func (b *base) load(data any) error {
	f, err := os.Open(b.ckptDir)
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
	if err != nil {
		return err
	}
	return nil
}
