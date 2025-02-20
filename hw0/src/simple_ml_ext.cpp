#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batchSize)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // Allocate memory for the logits and gradients
    float *Z = new float[batchSize * k];
    float *grad = new float[n * k];

    // update gradient for every batch
    for (size_t batchStart = 0; batchStart < m; batchStart += batchSize)
    {
        // 最后一个批次可能样本数不足 batchSize
        batchSize = std::min(batchSize, m - batchStart);

        // process batch
        std::memset(Z, 0, batchSize * k * sizeof(float));
        std::memset(grad, 0, n * k * sizeof(float));

        // compute Z = X_batch * theta
        for (size_t i = 0; i < batchSize; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                float sum = 0.0f;
                for (size_t p = 0; p < n; p++)
                {
                    sum += X[(batchStart + i) * n + p] * theta[p * k + j];
                }
                Z[i * k + j] = sum;
            }
        }

        // 对 Z 进行指数运算：Z = exp(Z)
        for (size_t i = 0; i < batchSize * k; i++)
        {
            Z[i] = std::exp(Z[i]);
        }

        // 3. 对 Z 每行归一化：每行除以该行元素之和
        for (size_t i = 0; i < batchSize; i++)
        {
            float rowSum = 0.0f;
            for (size_t j = 0; j < k; j++)
            {
                rowSum += Z[i * k + j];
            }
            for (size_t j = 0; j < k; j++)
            {
                Z[i * k + j] /= rowSum;
            }
        }

        // 4. 根据标签，对每个样本对应的位置减 1
        // 这里 y[batchStart + i] 是该样本的真实类别，要求在Z对应位置减1
        for (size_t i = 0; i < batchSize; i++)
        {   
            Z[i * k + y[batchStart + i]] -= 1.0f;
        }

        // 5. 计算梯度 grad = X_batch^T * Z
        // grad的形状为 (n, k)，即对于每个特征 p（0 <= p < n）和每个类别 j
        // grad[p][j] = sum_{i=0}^{batchSize-1} X[batchStart + i, p] * Z[i][j]
        for (size_t p = 0; p < n; p++)
        {
            for (size_t j = 0; j < k; j++)
            {
                float sum = 0.0f;
                for (size_t i = 0; i < batchSize; i++)
                {
                    sum += X[(batchStart + i) * n + p] * Z[i * k + j];
                }
                grad[p * k + j] = sum;
            }
        }

         // 6. 更新 theta: theta = theta - lr * grad / batchSize
        for (size_t p = 0; p < n * k; p++) {
            theta[p] -= lr * grad[p] / batchSize;
        }
    }

    // 释放内存
    delete[] Z;
    delete[] grad;
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def("softmax_regression_epoch_cpp", [](py::array_t<float, py::array::c_style> X, py::array_t<unsigned char, py::array::c_style> y, py::array_t<float, py::array::c_style> theta, float lr, int batch)
          { softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch); }, py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"), py::arg("batch"));
}
