package main

import (
	"fmt"

	Classifier "github.com/Yushgoel/CARTGo/classifier"
)

func main() {
	// var data [][]float64
	// data = append(data, []float64{1, 20, 3})
	// data = append(data, []float64{1, 2, 3})
	// data = append(data, []float64{-10, 20, 3})
	// data = append(data, []float64{-10, -20, -3})

	var tree Classifier.Node
	x_train, x_test, y_train, y_test := Classifier.Read_csv(713)
	tree = Classifier.Best_split(x_train, y_train, []int64{0, 1}, tree)

	// tree = Classifier.Best_split(data, []int64{0, 1, 1}, []int64{0, 1}, tree)

	fmt.Println(" ")
	fmt.Println(" ")
	Classifier.PrintTree(tree, "")
	fmt.Println(Classifier.Evaluate(tree, x_test, y_test))

}
