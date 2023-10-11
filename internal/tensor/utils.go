package tensor

func dup[T uint16 | int64](data []T) []T {
	ret := make([]T, len(data))
	copy(ret, data)
	return ret
}
