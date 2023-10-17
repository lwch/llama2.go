package checkpoint

type bfloat16Storage struct {
	base
}

var bf16Instance storage = &bfloat16Storage{}

func (*bfloat16Storage) New(ckptDir, fileName, location string, dataSize int) storage {
	var ret bfloat16Storage
	ret.init(ckptDir, fileName, location, dataSize)
	return &ret
}

func (f *bfloat16Storage) Load(offset int, count int64) (any, error) {
	data := make([]uint16, count)
	err := f.load(offset, data)
	return data, err
}

func (*bfloat16Storage) Type() storageType {
	return typeBFloat16
}
