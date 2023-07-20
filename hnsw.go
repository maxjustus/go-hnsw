package hnsw

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/maxjustus/go-hnsw/bitsetpool"
	"github.com/maxjustus/go-hnsw/distances"
	"github.com/maxjustus/go-hnsw/distqueue"
	"gonum.org/v1/gonum/mat"
)

const (
	CosineDist    = 1
	L2SquaredDist = 2
)

type node struct {
	sync.RWMutex
	locked  bool
	point   *mat.VecDense
	level   int
	friends [][]uint32
}

type Hnsw struct {
	sync.RWMutex
	M              int
	M0             int
	efConstruction int
	linkMode       int
	DelaunayType   int

	DistFunc func(*mat.VecDense, *mat.VecDense) float64
	DistType int

	nodes []node

	bitset *bitsetpool.BitsetPool

	LevelMult  float64
	maxLayer   int
	enterpoint uint32
}

// Load opens a index file previously written by Save(). Returnes a new index and the timestamp the file was written
func Load(filename string) (*Hnsw, int64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, err
	}
	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, 0, err
	}

	timestamp := readInt64(reader)

	h := new(Hnsw)
	h.M = readInt32(reader)
	h.M0 = readInt32(reader)
	h.efConstruction = readInt32(reader)
	h.linkMode = readInt32(reader)
	h.DelaunayType = readInt32(reader)
	h.LevelMult = readFloat64(reader)
	h.maxLayer = readInt32(reader)
	h.enterpoint = uint32(readInt32(reader))
	h.DistType = readInt32(reader)

	switch h.DistType {
	case CosineDist:
		h.DistFunc = distances.Cosine
	case L2SquaredDist:
		h.DistFunc = distances.L2Squared
	}

	h.bitset = bitsetpool.New()

	lenNodes := readInt32(reader)
	h.nodes = make([]node, lenNodes)

	for i := range h.nodes {

		var newVec mat.VecDense
		_, err := newVec.UnmarshalBinaryFrom(reader)

		if err != nil {
			panic(err)
		}

		h.nodes[i].point = &newVec

		h.nodes[i].level = readInt32(reader)

		numFriends := readInt32(reader)
		h.nodes[i].friends = make([][]uint32, numFriends)

		for j := range h.nodes[i].friends {
			friends := readInt32(reader)
			h.nodes[i].friends[j] = make([]uint32, friends)
			err = binary.Read(reader, binary.LittleEndian, h.nodes[i].friends[j])
			if err != nil {
				panic(err)
			}
		}

	}

	reader.Close()
	file.Close()

	return h, timestamp, nil
}

// Save writes to current index to a gzipped binary data file
func (h *Hnsw) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	writer := gzip.NewWriter(file)

	timestamp := time.Now().Unix()

	writeInt64(timestamp, writer)

	writeInt32(h.M, writer)
	writeInt32(h.M0, writer)
	writeInt32(h.efConstruction, writer)
	writeInt32(h.linkMode, writer)
	writeInt32(h.DelaunayType, writer)
	writeFloat64(h.LevelMult, writer)
	writeInt32(h.maxLayer, writer)
	writeInt32(int(h.enterpoint), writer)
	writeInt32(h.DistType, writer)

	lenNodes := len(h.nodes)
	writeInt32(lenNodes, writer)

	if err != nil {
		return err
	}
	for _, n := range h.nodes {

		//writeInt32(n.point.Len(), writer)

		_, err = n.point.MarshalBinaryTo(writer)

		if err != nil {
			panic(err)
		}
		writeInt32(n.level, writer)

		numFriends := len(n.friends)
		writeInt32(numFriends, writer)
		for _, friend := range n.friends {
			lenFriend := len(friend)
			writeInt32(lenFriend, writer)
			err = binary.Write(writer, binary.LittleEndian, friend)
			if err != nil {
				panic(err)
			}
		}
	}

	writer.Close()
	file.Close()

	return nil
}

func writeInt64(v int64, w io.Writer) {
	err := binary.Write(w, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
}

func writeInt32(v int, w io.Writer) {
	i := int32(v)
	err := binary.Write(w, binary.LittleEndian, &i)
	if err != nil {
		panic(err)
	}
}

func readInt32(r io.Reader) int {
	var i int32
	err := binary.Read(r, binary.LittleEndian, &i)
	if err != nil {
		panic(err)
	}
	return int(i)
}

func writeFloat64(v float64, w io.Writer) {
	err := binary.Write(w, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
}

func readInt64(r io.Reader) (v int64) {
	err := binary.Read(r, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
	return
}

func readFloat64(r io.Reader) (v float64) {
	err := binary.Read(r, binary.LittleEndian, &v)
	if err != nil {
		panic(err)
	}
	return
}

func (h *Hnsw) getFriends(n uint32, level int) []uint32 {
	if len(h.nodes[n].friends) < level+1 {
		return make([]uint32, 0)
	}
	return h.nodes[n].friends[level]
}

func (h *Hnsw) Link(first, second uint32, level int) {

	maxL := h.M
	if level == 0 {
		maxL = h.M0
	}

	h.RLock()
	node := &h.nodes[first]
	h.RUnlock()

	node.Lock()

	// check if we have allocated friends slices up to this level?
	if len(node.friends) < level+1 {
		for j := len(node.friends); j <= level; j++ {
			// allocate new list with 0 elements but capacity maxL
			node.friends = append(node.friends, make([]uint32, 0, maxL))
		}
		// now grow it by one and add the first connection for this layer
		node.friends[level] = node.friends[level][0:1]
		node.friends[level][0] = second

	} else {
		// we did have some already... this will allocate more space if it overflows maxL
		node.friends[level] = append(node.friends[level], second)
	}

	l := len(node.friends[level])

	if l > maxL {

		// to many links, deal with it

		switch h.DelaunayType {
		case 0:
			resultSet := distqueue.NewClosestLastQueue()

			for _, n := range node.friends[level] {
				resultSet.Push(n, h.DistFunc(node.point, h.nodes[n].point))
			}
			for resultSet.Size() > maxL {
				resultSet.Pop()
			}
			// FRIENDS ARE STORED IN DISTANCE ORDER, closest at index 0
			node.friends[level] = node.friends[level][0:maxL]
			for i := maxL - 1; i >= 0; i-- {
				item, exists := resultSet.Pop()
				if exists {
					node.friends[level][i] = item.ID
				}
			}

		case 1:

			resultSet := distqueue.NewClosestFirstQueue()

			for _, n := range node.friends[level] {
				resultSet.Push(n, h.DistFunc(node.point, h.nodes[n].point))
			}
			h.getNeighborsByHeuristicClosestFirst(resultSet, maxL)

			// FRIENDS ARE STORED IN DISTANCE ORDER, closest at index 0
			node.friends[level] = node.friends[level][0:maxL]
			for i := 0; i < maxL; i++ {
				item, exists := resultSet.Pop()
				if exists {
					node.friends[level][i] = item.ID
				}
			}
		}
	}
	node.Unlock()
}

func (h *Hnsw) getNeighborsByHeuristicClosestLast(resultSet1 distqueue.ItemQueue, M int) {
	if resultSet1.Size() <= M {
		return
	}

	resultSet := distqueue.NewClosestFirstQueue()
	tempList := distqueue.NewClosestLastQueue()
	result := make([]*distqueue.Item, 0, M)
	for resultSet1.Size() > 0 {
		item, exists := resultSet1.Pop()
		if exists {
			resultSet.PushItem(item)
		}
	}
	for resultSet.Size() > 0 {
		e, exists := resultSet.Pop()
		if !exists {
			break
		}

		good := true

		for _, r := range result {
			if h.DistFunc(h.nodes[r.ID].point, h.nodes[e.ID].point) < e.Distance {
				good = false
				break
			}
		}

		if good {
			result = append(result, &e)
		} else {
			tempList.PushItem(e)
		}
	}
	for len(result) < M && tempList.Size() > 0 {
		item, exists := tempList.Pop()
		if exists {
			result = append(result, &item)
		}
	}

	for _, item := range result {
		resultSet1.PushItem(*item)
	}
}

func (h *Hnsw) getNeighborsByHeuristicClosestFirst(resultSet distqueue.ItemQueue, M int) {
	if resultSet.Size() <= M {
		return
	}

	tempList := distqueue.NewClosestFirstQueue()
	result := make([]*distqueue.Item, 0, M)
	for resultSet.Size() > 0 {
		e, exists := resultSet.Pop()
		if !exists {
			break
		}

		good := true
		for _, r := range result {
			if h.DistFunc(h.nodes[r.ID].point, h.nodes[e.ID].point) < e.Distance {
				good = false
				break
			}
		}
		if good {
			result = append(result, &e)
		} else {
			tempList.PushItem(e)
		}
	}
	for len(result) < M && tempList.Size() > 0 {
		item, exists := tempList.Pop()
		if exists {
			result = append(result, &item)
		}
	}

	resultSet.Clear()

	for _, item := range result {
		resultSet.PushItem(*item)
	}
}

func New(M int, efConstruction int, first *mat.VecDense, metric string) (*Hnsw, error) {

	h := Hnsw{}
	h.M = M
	// default values used in c++ implementation
	h.LevelMult = 1 / math.Log(float64(M))
	h.efConstruction = efConstruction
	h.M0 = 2 * M
	h.DelaunayType = 1

	h.bitset = bitsetpool.New()

	switch metric {
	case "cosine":
		h.DistType = CosineDist
		h.DistFunc = distances.Cosine
	case "l2":
		h.DistType = L2SquaredDist
		h.DistFunc = distances.L2Squared
	default:
		return nil, errors.New(fmt.Sprintf("The metric %v is not implemented", metric))
	}

	// add first point, it will be our enterpoint (index 0)
	h.nodes = []node{node{level: 0, point: first}}

	return &h, nil
}

func (h *Hnsw) Stats() string {
	s := "HNSW Index\n"
	s = s + fmt.Sprintf("M: %v, efConstruction: %v\n", h.M, h.efConstruction)
	s = s + fmt.Sprintf("DelaunayType: %v\n", h.DelaunayType)
	s = s + fmt.Sprintf("Number of nodes: %v\n", len(h.nodes))
	s = s + fmt.Sprintf("Max layer: %v\n", h.maxLayer)
	memoryUseData := 0
	memoryUseIndex := 0
	levCount := make([]int, h.maxLayer+1)
	conns := make([]int, h.maxLayer+1)
	connsC := make([]int, h.maxLayer+1)
	for i := range h.nodes {
		levCount[h.nodes[i].level]++
		for j := 0; j <= h.nodes[i].level; j++ {
			if len(h.nodes[i].friends) > j {
				l := len(h.nodes[i].friends[j])
				conns[j] += l
				connsC[j]++
			}
		}
		pointMemSize := len(h.nodes[i].point.RawVector().Data) * 8
		memoryUseData += pointMemSize
		memoryUseIndex += h.nodes[i].level*h.M*4 + h.M0*4
	}
	for i := range levCount {
		avg := conns[i] / max(1, connsC[i])
		s = s + fmt.Sprintf("Level %v: %v nodes, average number of connections %v\n", i, levCount[i], avg)
	}
	s = s + fmt.Sprintf("Memory use for data: %v (%v bytes / point)\n", memoryUseData, memoryUseData/len(h.nodes))
	s = s + fmt.Sprintf("Memory use for index: %v (avg %v bytes / point)\n", memoryUseIndex, memoryUseIndex/len(h.nodes))
	return s
}

func (h *Hnsw) Grow(size int) {
	if size+1 <= len(h.nodes) {
		return
	}
	newNodes := make([]node, len(h.nodes), size+1)
	copy(newNodes, h.nodes)
	h.nodes = newNodes

}

func (h *Hnsw) Add(q *mat.VecDense, id uint32) {

	if id == 0 {
		panic("Id 0 is reserved, use ID:s starting from 1 when building index")
	}

	// generate random level
	curlevel := int(math.Floor(-math.Log(rand.Float64() * h.LevelMult)))

	epID := h.enterpoint
	currentMaxLayer := h.nodes[epID].level
	ep := &distqueue.Item{ID: h.enterpoint, Distance: h.DistFunc(h.nodes[h.enterpoint].point, q)}

	// assume Grow has been called in advance
	newID := id
	newNode := node{point: q, level: curlevel, friends: make([][]uint32, min(curlevel, currentMaxLayer)+1)}

	// first pass, find another ep if curlevel < maxLayer
	for level := currentMaxLayer; level > curlevel; level-- {
		changed := true
		for changed {
			changed = false
			for _, i := range h.getFriends(ep.ID, level) {
				d := h.DistFunc(h.nodes[i].point, q)
				if d < ep.Distance {
					ep = &distqueue.Item{ID: i, Distance: d}
					changed = true
				}
			}
		}
	}

	// second pass, ef = efConstruction
	// loop through every level from the new nodes level down to level 0
	// create new connections in every layer
	for level := min(curlevel, currentMaxLayer); level >= 0; level-- {

		resultSet := distqueue.NewClosestLastQueue()
		h.searchAtLayer(q, resultSet, h.efConstruction, ep, level)
		switch h.DelaunayType {
		case 0:
			// shrink resultSet to M closest elements (the simple heuristic)
			for resultSet.Size() > h.M {
				resultSet.Pop()
			}
		case 1:
			h.getNeighborsByHeuristicClosestLast(resultSet, h.M)
		}
		newNode.friends[level] = make([]uint32, resultSet.Size())
		for i := resultSet.Size() - 1; i >= 0; i-- {
			item, exists := resultSet.Pop()

			if exists {
				// store in order, closest at index 0
				newNode.friends[level][i] = item.ID
			}
		}
	}

	h.Lock()
	// Add it and increase slice length if neccessary
	if len(h.nodes) < int(newID)+1 {
		h.nodes = h.nodes[0 : newID+1]
	}
	h.nodes[newID] = newNode
	h.Unlock()

	// now add connections to newNode from newNodes neighbours (makes it visible in the graph)
	for level := min(curlevel, currentMaxLayer); level >= 0; level-- {
		for _, n := range newNode.friends[level] {
			h.Link(n, newID, level)
		}
	}

	h.Lock()
	if curlevel > h.maxLayer {
		h.maxLayer = curlevel
		h.enterpoint = newID
	}
	h.Unlock()
}

func (h *Hnsw) searchAtLayer(q *mat.VecDense, resultSet distqueue.ItemQueue, efConstruction int, ep *distqueue.Item, level int) {

	var pool, visited = h.bitset.Get()

	candidates := distqueue.NewClosestFirstQueue()

	visited.Set(uint(ep.ID))

	candidates.Push(ep.ID, ep.Distance)

	resultSet.Push(ep.ID, ep.Distance)

	for candidates.Size() > 0 {
		worstMatch, exists := resultSet.Peek() // worst distance so far
		if !exists {
			break
		}
		c, exists := candidates.Pop()

		if !exists || c.Distance > worstMatch.Distance {
			// since candidates is sorted, it wont get any better...
			break
		}

		if len(h.nodes[c.ID].friends) >= level+1 {
			friends := h.nodes[c.ID].friends[level]
			for _, n := range friends {
				if !visited.Test(uint(n)) {
					visited.Set(uint(n))
					d := h.DistFunc(q, h.nodes[n].point)
					topMatch, exists := resultSet.Peek()
					if !exists {
						break
					}

					if resultSet.Size() < efConstruction {
						item := resultSet.Push(n, d)
						candidates.PushItem(item)
					} else if topMatch.Distance > d {
						// keep length of resultSet to max efConstruction
						item, exists := resultSet.PopAndPush(n, d)
						if exists {
							candidates.PushItem(item)
						}
					}
				}
			}
		}
	}
	h.bitset.Free(pool)
}

// SearchBrute returns the true K nearest neigbours to search point q
func (h *Hnsw) SearchBrute(q *mat.VecDense, K int) distqueue.ItemQueue {
	resultSet := distqueue.NewClosestLastQueue()
	for i := 1; i < len(h.nodes); i++ {
		d := h.DistFunc(h.nodes[i].point, q)
		if resultSet.Size() < K {
			resultSet.Push(uint32(i), d)
			continue
		}
		topItem, exists := resultSet.Peek()
		if exists && d < topItem.Distance {
			resultSet.PopAndPush(uint32(i), d)
			continue
		}
	}
	return resultSet
}

// Benchmark test precision by comparing the results of SearchBrute and Search
func (h *Hnsw) Benchmark(q *mat.VecDense, ef int, K int) float64 {
	result := h.Search(q, ef, K)
	groundTruth := h.SearchBrute(q, K)
	truth := make([]uint32, 0)
	for groundTruth.Size() > 0 {
		item, exists := groundTruth.Pop()
		if exists {
			truth = append(truth, item.ID)
		}
	}
	p := 0
	for result.Size() > 0 {
		i, exists := result.Pop()
		if exists {
			for j := 0; j < K; j++ {
				if truth[j] == i.ID {
					p++
				}
			}
		}
	}
	return float64(p) / float64(K)
}

func (h *Hnsw) Search(q *mat.VecDense, ef int, K int) distqueue.ItemQueue {

	h.RLock()
	currentMaxLayer := h.maxLayer
	ep := &distqueue.Item{ID: h.enterpoint, Distance: h.DistFunc(h.nodes[h.enterpoint].point, q)}
	h.RUnlock()

	resultSet := distqueue.NewClosestLastQueue()
	// first pass, find best ep
	for level := currentMaxLayer; level > 0; level-- {
		changed := true
		for changed {
			changed = false
			for _, i := range h.getFriends(ep.ID, level) {
				d := h.DistFunc(h.nodes[i].point, q)
				if d < ep.Distance {
					ep.ID, ep.Distance = i, d
					changed = true
				}
			}
		}
	}
	h.searchAtLayer(q, resultSet, ef, ep, 0)

	for resultSet.Size() > K {
		resultSet.Pop()
	}
	return resultSet
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
