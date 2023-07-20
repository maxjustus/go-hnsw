package distqueue

import (
	"github.com/emirpasic/gods/queues/priorityqueue"
	"github.com/emirpasic/gods/utils"
)

// TODO: refactor to use gods priority queue package

type Item struct {
	ID       uint32
	Distance float64
}

type ItemQueue struct {
	*priorityqueue.Queue
}

func (queue *ItemQueue) Push(id uint32, similarity float64) Item {
	item := Item{id, similarity}
	queue.Enqueue(item)
	return item
}

func (queue ItemQueue) Peek() (Item, bool) {
	item, exists := queue.Queue.Peek()
	return item.(Item), exists
}

func (queue ItemQueue) Values() []Item {
	vals := queue.Queue.Values()

	items := make([]Item, len(vals))
	for i, val := range vals {
		items[i] = val.(Item)
	}

	return items
}

func (queue *ItemQueue) PushItem(item Item) {
	queue.Enqueue(item)
}

func (queue *ItemQueue) Pop() (Item, bool) {
	item, exists := queue.Dequeue()
	return item.(Item), exists
}

func (queue ItemQueue) PopAndPush(id uint32, similarity float64) (Item, bool) {
	item := Item{id, similarity}
	oldItem, exists := queue.Pop()
	queue.PushItem(item)
	return oldItem, exists
}

func byDistance(a, b interface{}) int {
	return utils.Float64Comparator( // Note "-" for descending order
		a.(Item).Distance,
		b.(Item).Distance,
	)
}

func byReverseDistance(a, b interface{}) int {
	return -utils.Float64Comparator( // Note "-" for descending order
		a.(Item).Distance,
		b.(Item).Distance,
	)
}

func NewClosestFirstQueue() ItemQueue {
	return ItemQueue{priorityqueue.NewWith(byDistance)}
}

func NewClosestLastQueue() ItemQueue {
	return ItemQueue{priorityqueue.NewWith(byReverseDistance)}
}
