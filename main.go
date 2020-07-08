package main

import (
	"fmt"

	Classifier "github.com/Yushgoel/CARTGo/classifier"
)

func main() {

	x_train, x_test, y_train, y_test := Classifier.Read_csv(713)

	fmt.Println(" ")
	fmt.Println("With sklearn Interface")

	decTree := Classifier.DecisionTreeClassifier("entropy", -1, []int64{0, 1})
	decTree = Classifier.Fit(decTree, x_train, y_train)
	fmt.Println(" ")
	fmt.Println(" ")
	Classifier.PrintTree(decTree)
	fmt.Println(Classifier.Evaluate(decTree, x_test, y_test))
	fmt.Println(Classifier.Predict(decTree, x_test))
}
