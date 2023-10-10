package checkpoint

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type bfloat16 struct {
	base
	data []uint16
}

var _ storage = &bfloat16{}

func (*bfloat16) New(wg *sync.WaitGroup, size int, file *zip.File) (storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("BFloat16.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret bfloat16
	ret.data = make([]uint16, size)
	wg.Add(1)
	go func() {
		defer wg.Done()
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("BFloat16.New: can not read file %s: %v", file.Name, err))
		}
	}()
	return &ret, nil
}

func (f *bfloat16) Get() interface{} {
	return f.data
}

func (*bfloat16) Type() storageType {
	return typeBFloat16
}
