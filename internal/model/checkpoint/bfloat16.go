package checkpoint

type bfloat16 struct {
	base
}

var bf16Instance Storage = &bfloat16{}

func (*bfloat16) New(ckptDir, fileName string, dataSize int) Storage {
	var ret bfloat16
	ret.init(ckptDir, fileName, dataSize)
	return &ret
}

func (f *bfloat16) Load() (any, error) {
	data := make([]uint16, f.dataSize)
	err := f.load(data)
	return data, err
}

func (*bfloat16) Type() storageType {
	return typeBFloat16
}
