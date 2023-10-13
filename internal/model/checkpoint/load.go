package checkpoint

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"

	"github.com/nlpodyssey/gopickle/pickle"
	"github.com/nlpodyssey/gopickle/types"
)

type findClassFunc func(module, name string) (interface{}, error)

type Model struct {
	ckptDir  string
	storages map[string]storage
	files    map[string]string
	params   map[string]*Tensor
}

func Load(dir string) (*Model, error) {
	f, err := os.Open(dir)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	zr, err := zip.NewReader(f, fi.Size())
	if err != nil {
		return nil, err
	}
	data, err := getDataPkl(zr)
	if err != nil {
		return nil, err
	}
	defer data.Close()
	pkl := pickle.NewUnpickler(data)
	var m Model
	m.ckptDir = dir
	m.storages = make(map[string]storage)
	m.files = files(zr)
	pkl.FindClass = m.buildFindClass(pkl.FindClass)
	pkl.PersistentLoad = m.persistentLoad
	params, err := pkl.Load()
	if err != nil {
		return nil, err
	}
	err = m.loadParams(params)
	if err != nil {
		return nil, err
	}
	return &m, nil
}

func getDataPkl(r *zip.Reader) (io.ReadCloser, error) {
	for _, f := range r.File {
		if filepath.Base(f.Name) == "data.pkl" {
			return f.Open()
		}
	}
	return nil, errors.New("data.pkl not found")
}

func files(r *zip.Reader) map[string]string {
	result := make(map[string]string)
	for _, file := range r.File {
		_, recordName := path.Split(file.Name)
		result[recordName] = file.Name
	}
	return result
}

func (m *Model) buildFindClass(cb findClassFunc) findClassFunc {
	return func(module, name string) (interface{}, error) {
		switch module + "." + name {
		case "torch._utils._rebuild_tensor_v2":
			return &rebuildTensorV2{}, nil
		case "torch.BFloat16Storage": // bfloat16
			return bf16Instance, nil
		case "torch.FloatStorage": // float32
			return f32Instance, nil
		default:
			if cb == nil {
				return nil, fmt.Errorf("class not found: %s %s", module, name)
			}
			return cb(module, name)
		}
	}
}

func (m *Model) persistentLoad(id interface{}) (interface{}, error) {
	tuple, tupleOk := id.(*types.Tuple)
	if !tupleOk || tuple.Len() == 0 {
		return nil, fmt.Errorf("PersistentLoad: non-empty tuple expected, got %#v", id)
	}
	typename, typenameOk := tuple.Get(0).(string)
	if !typenameOk {
		return nil, fmt.Errorf("PersistentLoad: cannot get typename")
	}
	if typename != "storage" {
		return nil, fmt.Errorf("unknown typename for PersistentLoad, expected 'storage' but got '%s'", typename)
	}
	if tuple.Len() < 5 {
		return nil, fmt.Errorf("PersistentLoad: unexpected storage data length")
	}
	storageType, storageTypeOk := tuple.Get(1).(storage)
	key, keyOk := tuple.Get(2).(string)
	location, locationOk := tuple.Get(3).(string)
	size, sizeOk := tuple.Get(4).(int)
	if !storageTypeOk || !keyOk || !locationOk || !sizeOk {
		return nil, fmt.Errorf("PersistentLoad: unexpected data types")
	}

	file, ok := m.files[key]
	if !ok {
		return nil, fmt.Errorf("PersistentLoad: file not found: %s", key)
	}

	storage, ok := m.storages[key]
	if !ok {
		storage = storageType.New(m.ckptDir, file, location, size)
		m.storages[key] = storage
	}
	return storage, nil
}

func (m *Model) loadParams(params interface{}) error {
	switch params.(type) {
	case *types.OrderedDict:
		return m.loadParamsOrderedDict(params)
	case *types.Dict:
		return m.loadParamsDict(params)
	default:
		return fmt.Errorf("loadParams: unexpected data type: %#v", params)
	}
}

func (m *Model) loadParamsOrderedDict(params interface{}) error {
	dict := params.(*types.OrderedDict)
	m.params = make(map[string]*Tensor)
	for _, entry := range dict.Map {
		key, keyOk := entry.Key.(string)
		value, valueOk := entry.Value.(*Tensor)
		if !keyOk || !valueOk {
			return fmt.Errorf("loadParamsOrderedDict: unexpected data types, key: %#v, value: %#v", key, value)
		}
		m.params[key] = value
	}
	return nil
}

func (m *Model) loadParamsDict(params interface{}) error {
	dict := params.(*types.Dict)
	m.params = make(map[string]*Tensor)
	for _, entry := range *dict {
		key, keyOk := entry.Key.(string)
		value, valueOk := entry.Value.(*Tensor)
		if !keyOk || !valueOk {
			return fmt.Errorf("loadParamsDict: unexpected data types, key: %#v, value: %#v", key, value)
		}
		m.params[key] = value
	}
	return nil
}

func (m *Model) Params() map[string]*Tensor {
	return m.params
}
