package param

func Quantize(values []float32, t Type, groupSize int64) ([]byte, []byte) {
	switch t {
	case TypeQI8:
		return quantizeInt8(values, groupSize)
	}
	return nil, nil
}
