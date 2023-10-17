package checkpoint

import (
	"archive/zip"
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"reflect"
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
	New(ckptDir, fileName, location string, dataSize int) storage
	Type() storageType
	Load(offset int, count int64) (any, error)
}

type base struct {
	ckptDir  string
	fileName string
	location string
	count    int
}

func (b *base) init(ckptDir, fileName, location string, count int) {
	b.ckptDir = ckptDir
	b.fileName = fileName
	b.location = location
	b.count = count
}

func (b *base) load(offset int, data any) error {
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
	if reflect.TypeOf(data).Kind() != reflect.Slice {
		return fmt.Errorf("data is not a slice")
	}
	r := bufio.NewReader(fs)
	skip := offset * int(reflect.TypeOf(data).Elem().Size())
	for skip > 0 {
		n, err := r.Discard(skip)
		if err != nil {
			return err
		}
		skip -= n
	}
	switch data.(type) {
	case []uint16, []float32:
	default:
		return fmt.Errorf("unsupported data type: %T", data)
	}
	err = binary.Read(r, binary.LittleEndian, data)
	if err != nil {
		return err
	}
	return nil
}
