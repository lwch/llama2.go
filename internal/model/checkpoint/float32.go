package checkpoint

type float32Storage struct {
	base
}

var f32Instance storage = &float32Storage{}

func (*float32Storage) New(ckptDir, fileName, location string, dataSize int) storage {
	var ret float32Storage
	ret.init(ckptDir, fileName, location, dataSize)
	return &ret
}

func (f *float32Storage) Load(offset int, count int64) (any, error) {
	data := make([]float32, count)
	err := f.load(offset, data)
	return data, err
}

func (*float32Storage) Type() storageType {
	return typeFloat
}
