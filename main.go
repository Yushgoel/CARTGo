package main

import (
	"fmt"
	"math"

	Regressor "github.com/Yushgoel/CARTGO/regressor"
	Classifier "github.com/Yushgoel/CARTGo/classifier"
	"github.com/sjwhitworth/golearn/base"
)

func main() {

	x_train, x_test, y_train, y_test := Classifier.Read_csv(713)

	fmt.Println(" ")
	fmt.Println("With sklearn Interface")

	decTree := Classifier.DecisionTreeClassifier("entropy", -1, []int64{0, 1})
	decTree = Classifier.Fit(decTree, x_train, y_train)
	fmt.Println(" ")
	fmt.Println(" ")
	// Classifier.PrintTree(decTree)
	fmt.Println(Classifier.Evaluate(decTree, x_test, y_test))

	//fmt.Println(Classifier.Predict(decTree, x_test))
	// fmt.Println(Regressor.Mean_absolute_error([]float64{1.0, 1.0, 1.0}, 19.0))
	// x_train_reg, x_test_reg, y_train_reg, y_test_reg := Regressor.Read_csv(1169)
	// fmt.Println(len(x_train_reg), len(x_test_reg), len(y_train_reg), len(y_test_reg))
	reg_tree := Regressor.DecisionTreeRegressor("mse", -1)
	// reg_tree = Regressor.Fit(reg_tree, x_train_reg, y_train_reg)
	// predictions := Regressor.Predict(reg_tree, x_test_reg)
	// Regressor.PrintTree(reg_tree)
	// fmt.Println(mae(y_test_reg, predictions))
	rawData, _ := base.ParseCSVToInstances("regressor_data.csv", false)

	train := convertInstancesToProblemVec(rawData)
	y := convertInstancesToLabelVec(rawData)
	fmt.Println(train)
	fmt.Println(y)
	reg_tree = Regressor.Fit(reg_tree, train, y)
	fmt.Println(mae(y, Regressor.Predict(reg_tree, train)))
	// fmt.Println(x_train_reg, y_train_reg)
}

func mae(y []float64, y_pred []float64) float64 {
	error := 0.0
	for i := range y {
		error += math.Abs(y[i] - y_pred[i])
	}
	error /= float64(len(y))
	return error
}

func convertInstancesToProblemVec(X base.FixedDataGrid) [][]float64 {
	// Allocate problem array
	_, rows := X.Size()
	problemVec := make([][]float64, rows)

	// Retrieve numeric non-class Attributes
	numericAttrs := base.NonClassFloatAttributes(X)
	numericAttrSpecs := base.ResolveAttributes(X, numericAttrs)

	// Convert each row
	X.MapOverRows(numericAttrSpecs, func(row [][]byte, rowNo int) (bool, error) {
		// Allocate a new row
		probRow := make([]float64, len(numericAttrSpecs))
		// Read out the row
		for i, _ := range numericAttrSpecs {
			probRow[i] = base.UnpackBytesToFloat(row[i])
		}
		// Add the row
		problemVec[rowNo] = probRow
		return true, nil
	})
	return problemVec
}
func convertInstancesToLabelVec(X base.FixedDataGrid) []float64 {
	// Get the class Attributes
	classAttrs := X.AllClassAttributes()
	// Only support 1 class Attribute
	if len(classAttrs) != 1 {
		panic(fmt.Sprintf("%d ClassAttributes (1 expected)", len(classAttrs)))
	}
	// ClassAttribute must be numeric
	if _, ok := classAttrs[0].(*base.FloatAttribute); !ok {
		panic(fmt.Sprintf("%s: ClassAttribute must be a FloatAttribute", classAttrs[0]))
	}
	// Allocate return structure
	_, rows := X.Size()
	labelVec := make([]float64, rows)
	// Resolve class Attribute specification
	classAttrSpecs := base.ResolveAttributes(X, classAttrs)
	X.MapOverRows(classAttrSpecs, func(row [][]byte, rowNo int) (bool, error) {
		labelVec[rowNo] = base.UnpackBytesToFloat(row[0])
		return true, nil
	})
	return labelVec
}
