package cmd

import (
	"os"
	"runtime/pprof"
	"time"

	"github.com/lwch/runtime"
)

func profile() {
	f, err := os.Create("cpu.pprof")
	runtime.Assert(err)
	defer f.Close()
	runtime.Assert(pprof.StartCPUProfile(f))
	time.Sleep(time.Minute)
	pprof.StopCPUProfile()
	f.Close()
	os.Exit(0)
}
