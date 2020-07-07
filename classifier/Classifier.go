package classifier

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	Sorter "github.com/Yushgoel/CARTGo/sorter"
)

func Gini_impurity(y []int64, labels []int64) (float64, int64) {
	n_instances := len(y)
	gini := 0.0
	max_label_count := 0
	var max_label int64 = 0
	for label := range labels {
		num_label := 0
		for target := range y {
			if y[target] == labels[label] {
				num_label++
			}
		}
		p := float64(num_label) / float64(n_instances)
		gini += p * (1 - p)
		if num_label > max_label_count {
			max_label = labels[label]
			max_label_count = num_label
		}
	}
	return gini, max_label
}

func Entropy(y []int64, labels []int64) (float64, int64) {
	n_instances := len(y)
	entropy := 0.0
	max_label_count := 0
	var max_label int64 = 0
	for label := range labels {
		num_label := 0
		for target := range y {
			if y[target] == labels[label] {
				num_label++
			}
		}
		p := float64(num_label) / float64(n_instances)

		log_p := math.Log2(p)
		if p == 0 {
			log_p = 0
		}
		entropy += -p * log_p
		if num_label > max_label_count {
			max_label = labels[label]
			max_label_count = num_label
		}
	}
	return entropy, max_label
}

func Test_split(data [][]float64, feature int64, y []int64, threshold float64) ([][]float64, [][]float64, []int64, []int64) {
	var left [][]float64
	var right [][]float64
	var lefty []int64
	var righty []int64

	for i := range data {
		example := data[i]
		if example[feature] < threshold {
			left = append(left, example)
			lefty = append(lefty, y[i])
		} else {
			right = append(right, example)
			righty = append(righty, y[i])
		}
	}

	return left, right, lefty, righty
}

func stringInSlice(a float64, list []float64) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

func Find_unique(data []float64) []float64 {
	var unique []float64
	for i := range data {
		if !stringInSlice(data[i], unique) {
			unique = append(unique, data[i])
		}
	}
	return unique
}

func Get_feature(data [][]float64, feature int64) []float64 {
	var feature_vals []float64
	for i := range data {
		feature_vals = append(feature_vals, data[i][feature])
	}
	return feature_vals
}

type Node struct {
	Left        *Node
	Right       *Node
	Threshold   float64
	Feature     int64
	Left_Label  int64
	Right_Label int64
	Use_not     bool
}

var Tried_splits [][]float64

func Validate(feature int64, threshold float64) bool {
	for i := range Tried_splits {
		split := Tried_splits[i]
		feature_tried, threshold_tried := split[0], split[1]
		if int64(feature_tried) == feature && threshold_tried == threshold {
			return false
		}
	}
	return true
}

func ReOrder_data(feature_val []float64, data [][]float64, y []int64) ([][]float64, []int64) {
	s := Sorter.NewSlice(feature_val)
	sort.Sort(s)

	indexes := s.Idx

	var data_sorted [][]float64
	var y_sorted []int64

	for _, index := range indexes {
		data_sorted = append(data_sorted, data[index])
		y_sorted = append(y_sorted, y[index])
	}

	return data_sorted, y_sorted

}

func Update_split(left [][]float64, lefty []int64, right [][]float64, righty []int64, feature int64, threshold float64) ([][]float64, []int64, [][]float64, []int64) {

	for right[0][feature] < threshold {
		left = append(left, right[0])
		right = right[1:]
		lefty = append(lefty, righty[0])
		righty = righty[1:]
	}

	return left, lefty, right, righty
}

func sum(y []int64) int64 {
	var sum_ int64 = 0
	for i := range y {
		sum_ += y[i]
	}
	return sum_
}

// Essentially the Fit Method
func Best_split(data [][]float64, y []int64, labels []int64, upperNode Node, criterion string, maxDepth int64, depth int64) Node {
	criterion = strings.ToLower(criterion)

	depth++

	if depth > maxDepth {
		return upperNode
	}

	num_features := len(data[0])
	var best_gini float64
	var orig_gini float64

	if criterion == "gini" {
		orig_gini, upperNode.Left_Label = Gini_impurity(y, labels)
	} else if criterion == "entropy" {
		orig_gini, upperNode.Left_Label = Entropy(y, labels)
	} else {
		panic("Please enter a valid impurity function from GINI or ENTROPY")
	}

	best_gini = orig_gini

	best_left := data
	best_right := data
	best_lefty := y
	best_righty := y

	num_data := len(data)

	best_left_gini := best_gini
	best_right_gini := best_gini

	upperNode.Use_not = true

	var leftN Node
	var rightN Node
	// Iterate over all features
	for i := 0; i < num_features; i++ {
		feature_val := Get_feature(data, int64(i))
		unique := Find_unique(feature_val)
		sort.Float64s(unique)
		num_unique := len(unique)

		sort_data, sort_y := ReOrder_data(feature_val, data, y)

		first_time := true

		var left, right [][]float64
		var lefty, righty []int64

		for j := range unique {
			if j != (num_unique - 1) {
				threshold := (unique[j] + unique[j+1]) / 2
				if Validate(int64(i), threshold) {
					if first_time {
						left, right, lefty, righty = Test_split(sort_data, int64(i), sort_y, threshold)
						first_time = false
					} else {
						left, lefty, right, righty = Update_split(left, lefty, right, righty, int64(i), threshold)
					}

					var left_gini float64
					var right_gini float64
					var left_labels int64
					var right_labels int64

					if criterion == "gini" {
						left_gini, left_labels = Gini_impurity(lefty, labels)
						right_gini, right_labels = Gini_impurity(righty, labels)
					} else if criterion == "entropy" {
						left_gini, left_labels = Entropy(lefty, labels)
						right_gini, right_labels = Entropy(righty, labels)
					} else {

					}
					sub_gini := (left_gini * float64(len(left)) / float64(num_data)) + (right_gini * float64(len(right)) / float64(num_data))

					if sub_gini < best_gini {
						best_gini = sub_gini
						best_left = left
						best_right = right
						best_lefty = lefty
						best_righty = righty
						upperNode.Threshold = threshold
						upperNode.Feature = int64(i)

						upperNode.Left_Label = left_labels
						upperNode.Right_Label = right_labels

						best_left_gini = left_gini
						best_right_gini = right_gini
					}
				}

			}
		}
	}

	if best_gini == orig_gini {
		upperNode.Use_not = false
		return upperNode
	}

	if best_gini > 0 {

		if best_left_gini > 0 {
			Tried_splits = append(Tried_splits, []float64{float64(upperNode.Feature), upperNode.Threshold})
			leftN = Best_split(best_left, best_lefty, labels, leftN, criterion, maxDepth, depth)
			if leftN.Use_not == true {
				upperNode.Left = &leftN
			}

		}
		if best_right_gini > 0 {
			Tried_splits = append(Tried_splits, []float64{float64(upperNode.Feature), upperNode.Threshold})
			rightN = Best_split(best_right, best_righty, labels, rightN, criterion, maxDepth, depth)
			if rightN.Use_not == true {
				upperNode.Right = &rightN
			}

		}
		// print("Threshold " + upperNode.Threshold)

	}
	// fmt.Print("IGNORE THIS LINE")
	// fmt.Println(best_left, best_right, best_lefty, best_righty)
	return upperNode
}

func PrintTree(tree Node, spacing string) float64 {

	fmt.Print(spacing + "Feature ")
	fmt.Print(tree.Feature)
	fmt.Print(" < ")
	fmt.Println(tree.Threshold)

	if tree.Left == nil {
		fmt.Println(spacing + "---> True")
		fmt.Print("  " + spacing + "PREDICT    ")
		fmt.Println(tree.Left_Label)
	}
	if tree.Right == nil {
		fmt.Println(spacing + "---> FALSE")
		fmt.Print("  " + spacing + "PREDICT    ")
		fmt.Println(tree.Right_Label)
	}

	if tree.Left != nil {
		fmt.Println(spacing + "---> True")
		PrintTree(*tree.Left, spacing+"  ")
	}

	if tree.Right != nil {
		fmt.Println(spacing + "---> False")
		PrintTree(*tree.Right, spacing+"  ")
	}

	return 0.0
}

func Predict_single(tree Node, instance []float64) int64 {
	if instance[tree.Feature] < tree.Threshold {
		if tree.Left == nil {
			return tree.Left_Label
		} else {
			return Predict_single(*tree.Left, instance)
		}
	} else {
		if tree.Right == nil {
			return tree.Right_Label
		} else {
			return Predict_single(*tree.Right, instance)
		}
	}
}

func Predict(tree Node, test [][]float64) []int64 {
	var preds []int64
	for i := range test {
		i_pred := Predict_single(tree, test[i])
		preds = append(preds, i_pred)
	}
	return preds
}

func Read_csv(train_size int) ([][]float64, [][]float64, []int64, []int64) {
	var x_train [][]float64
	var x_test [][]float64

	var y_train []int64
	var y_test []int64

	csvfile, err := os.Open("/Users/yush/go/src/github.com/YushGoel/CARTGo/data.csv")
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	// Parse the file
	r := csv.NewReader(csvfile)
	//r := csv.NewReader(bufio.NewReader(csvfile))
	counter := 0
	// Iterate through the records
	for {
		// Read each record from csv
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		if counter == 0 {
			fmt.Println("Reading Data")
		} else {
			var row []float64
			for i := 1; i < 2; i++ {
				record, _ := strconv.ParseFloat(record[i], 8)
				row = append(row, record)
			}
			y, _ := strconv.ParseFloat(record[0], 8)
			if counter < train_size {
				x_train = append(x_train, row)
				y_train = append(y_train, int64(y))
			} else {
				x_test = append(x_test, row)
				y_test = append(y_test, int64(y))
			}
		}
		counter++
	}

	return x_train, x_test, y_train, y_test
}

func Evaluate(tree Node, x_test [][]float64, y_test []int64) float64 {
	preds := Predict(tree, x_test)
	accuracy := 0.0
	for i := range preds {
		if preds[i] == y_test[i] {
			accuracy++
		}
	}
	accuracy /= float64(len(y_test))
	return accuracy
}
