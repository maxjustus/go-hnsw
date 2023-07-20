package distqueue

import (
	"math/rand"
	"testing"
)

func TestClosestFirstQueue(t *testing.T) {
	pq := NewClosestFirstQueue()

	for i := 0; i < 10; i++ {
		pq.Enqueue(Item{rand.Uint32(), rand.Float64()})
	}

	t.Log("Closest first, pop")
	item, _ := pq.Pop()

	t.Logf("TOP before first top: %v", item)

	var l float64 = 0.0
	for pq.Size() > 0 {
		item, exists := pq.Pop()
		if !exists {
			break
		}

		similarity := item.Distance

		if similarity < l {
			t.Error("Incorrect order")
		}

		l = similarity
		t.Logf("%+v", item)
	}
}

func TestClosestLastQueue(t *testing.T) {
	pq2 := NewClosestLastQueue()

	var l = 1.0

	for i := 0; i < 10; i++ {
		pq2.Push(rand.Uint32(), rand.Float64())
	}

	t.Log("Closest last, pop")
	for !pq2.Empty() {
		item, exists := pq2.Pop()

		if !exists {
			break
		}

		similarity := item.Distance

		if similarity > l {
			t.Error("Incorrect order")
		}

		l = similarity
		t.Logf("%+v", item)
	}
}

func TestKBest(t *testing.T) {

	pq := NewClosestFirstQueue()

	for i := 0; i < 20; i++ {
		pq.Push(rand.Uint32(), rand.Float64())
	}

	// return K best matches, ordered as best first
	t.Log("closest last, still return K best")
	K := 10
	for pq.Size() > K {
		pq.Dequeue()
	}

	res := make([]Item, K)
	for i := K - 1; i >= 0; i-- {
		item, exists := pq.Pop()
		if !exists {
			break
		}

		res[i] = item
	}

	for i := 0; i < len(res); i++ {
		t.Logf("%+v", res[i])
	}
}
