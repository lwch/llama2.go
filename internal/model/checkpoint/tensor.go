package checkpoint

type Tensor struct {
	st     storage
	offset int
	size   []int64
	stride []int64
}

func (t *Tensor) Load() (any, error) {
	count := t.size[0]
	for i := 1; i < len(t.size); i++ {
		count *= t.size[i]
	}
	return t.st.Load(t.offset, count)
}

func (t *Tensor) Shape() []int64 {
	return t.size
}
