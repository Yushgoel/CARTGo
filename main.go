package main

import (
	"fmt"

	Classifier "github.com/Yushgoel/CARTGo/classifier"
)

func main() {
	// For test cases

	// var data [][]float64
	// data = append(data, []float64{5, 20, 3})
	// data = append(data, []float64{3, 20, 3})
	// data = append(data, []float64{3, 20, 3})
	// data = append(data, []float64{1, 20, 3})
	// data = append(data, []float64{2, 20, 3})
	// data = append(data, []float64{1, 2, 3})

	// feature_0 := Classifier.Get_feature(data, 0)
	// fmt.Println(Classifier.ReOrder_data(feature_0, data, []int64{0, 1, 1, 0, 1, 0}))
	var tree Classifier.Node

	// tree = Classifier.Best_split(data, []int64{0, 1, 0}, []int64{0, 1}, tree)

	x_train, x_test, y_train, y_test := Classifier.Read_csv(713)
	tree = Classifier.Best_split(x_train, y_train, []int64{0, 1}, tree, 5, 0)

	fmt.Println(" ")
	fmt.Println(" ")
	Classifier.PrintTree(tree, "")
	fmt.Println(Classifier.Evaluate(tree, x_test, y_test))

}
