package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func sigmoid(v mat.Matrix) *mat.Dense {
	rows, cols := v.Dims()
	matrixForExp := mat.NewDense(rows, cols, nil)

	for i:=0; i<rows; i++{
		for j:=0; j<cols; j++{
			cellAfterExp := 1.0/(1.0+math.Exp(v.At(i,j)))
			//fmt.Printf("Cell after exp: %f\n", cellAfterExp)
			matrixForExp.Set(i,j,cellAfterExp)
		}
	}

	return matrixForExp
}
func main() {
	//Входные данные
	trainingInputs := mat.NewDense(5, 3, []float64{0,0,1,1,1,1,1,0,1,0,1,1,0,1,0})
	//println("Training inputs:")
	//matPrint(trainingInputs)
	//fmt.Println()

	//Выходные данные
	trainingOutputs := mat.NewDense(5, 1, []float64{0,1,1,0,0})
	//println("Training outputs:")
	//matPrint(trainingOutputs)
	//fmt.Println()

	//Генератор случайных чисел
	rand.Seed(time.Now().UnixNano())

	//Инициализация синаптических весов
	////Объявление и инициализация пустой матрицы
	v := make([]float64, 3)
	for i := 0; i < 3; i++ {
		v[i] = float64(0)
	}
	synapticWeights := mat.NewDense(3, 1, v)

	////Заполнение матрицы
	for i:=0; i<3; i++{
		arg := 2*rand.Float64()-1
		synapticWeights.Set(i, 0, arg)
	}
	//println("Synaptic weights:")
	//matPrint(synapticWeights)
	//println()

	//Тренировка "нейросети"
	////Количество итераций для тренировки
	circles := 5000
	outputs := mat.NewDense(5, 1, nil) //Пустая матрица

	for i := 0; i<circles; i++ {
		inputData := trainingInputs


		////Перемножение матриц
		outputs.Product(inputData, synapticWeights)
		//println("Outputs after prod:")
		//matPrint(outputs)
		//println()

		outputs = sigmoid(outputs)
		//println("Outputs after sigmoid:")
		//matPrint(outputs)
		//println()


		////Нахождение ошибки
		err := mat.NewDense(5, 1, nil) //Пустая матрица
		err.Sub(trainingOutputs, outputs)
		//println("Err after prod & sigmoid:")
		//matPrint(err)
		//println()

		////Корректирование синаптических весов
		tempOutputs := mat.NewDense(5, 1, nil) //Пустая матрица
		//////Единичная матрица
		singleMatrix:= mat.NewDense(5, 1, nil)
		singleMatrixRow, singleMatrixCol := singleMatrix.Dims()
		for i := 0; i<singleMatrixRow; i++{
			for j := 0; j<singleMatrixCol; j++{
				singleMatrix.Set(i,j,1)
			}
		}
		//println("Single matrix:")
		//matPrint(singleMatrix)
		//println()

		//////Временная матрица для выходных данных
		tempOutputs.Sub(singleMatrix, outputs)
		//println("Temporary outputs matrix:")
		//matPrint(tempOutputs)
		//println()

		temp1 := mat.NewDense(5, 5, nil)
		temp1.Product(tempOutputs, outputs.T())
		//println("Temp2:")
		//matPrint(temp1)
		//println()
		temp2 := mat.NewDense(5,1,nil)
		temp2.Product(temp1, err)
		//println("Temp2:")
		//matPrint(temp2)
		//println()

		//////Корректирование весов
		adjustments := mat.NewDense(3, 1, nil)
		adjustments.Product(inputData.T(), temp2)
		//println("Adjustments:")
		//matPrint(adjustments)
		//println()

		synapticWeights.Add(adjustments, synapticWeights)
		//println("Synaptic weights at this circle:")
		//matPrint(synapticWeights)
		//println()
	}
	println("Result: ")
	matPrint(outputs)
}
