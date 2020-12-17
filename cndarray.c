#include "stdio.h"
#include "math.h"
#include "limits.h"
#include <python3.9/Python.h>
#include <python3.9/structmember.h>



static double *new_double(size_t size) {
    return (double*)malloc(sizeof(double) * size);
}
static size_t *new_size_t(size_t size) {
    return (size_t*)malloc(sizeof(size_t) * size);
}
static int *new_int(size_t size) {
    return (int*)malloc(sizeof(int) * size);
}
static size_t* reverse(size_t* array1, size_t size) {
    size_t *array2 = new_size_t(size);
    for (size_t i = 0; i < size; ++i) {
        array2[i] = array1[size - i - 1];
    }
    return array2;
}
static double __add__(double double1, double double2) {
    return double1 + double2;
}
static double __sub__(double double1, double double2) {
    return double1 - double2;
}
static double __mul__(double double1, double double2) {
    return double1 * double2;
}
static double __div__(double double1, double double2) {
    return double1 / double2;
}
static double __pow__(double double1, double double2) {
    return pow(double1, double2);
}
static double sigmoid(double double1) {
    return 1 / (1 + exp(-double1));
}
static double d_sigmoid(double double1) {
    return sigmoid(double1) * (1 - sigmoid(double1));
}
static double d_tanh(double double1) {
    return 1 - pow(tanh(double1), 2);
}
static double relu(double double1) {
    if (double1 > 0)
        return double1;
    return 0;
}
static double d_relu(double double1) {
    if (double1 > 0)
        return 1;
    return 0;
}



typedef struct {
    PyObject_HEAD
    double *data;
    size_t *shape;
    size_t size;
    size_t dims;
} ArrayObject;

static PyTypeObject ArrayType;
static PyTypeObject *ArrayTypePtr = &ArrayType;
static PyObject * Array_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    ArrayObject *self;
    self = (ArrayObject*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static int Array_init(ArrayObject *self, PyObject *args, PyObject *kwargs)
{
    static char* kwargs_list[] = {"data", "shape", NULL};
    PyObject *data = NULL, *shape = NULL;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OO", kwargs_list,
            &data, &shape)) { return -1; }
    self->size = PyList_Size(data);
    self->dims = PyList_Size(shape);
    self->data = new_double(self->size);
    if (self->dims != 0) {
        self->shape = new_size_t(self->dims);
    }
    for (size_t i = 0; i < self->size; ++i)
        self->data[i] = PyFloat_AsDouble(PyList_GET_ITEM(data, i));
    for (size_t i = 0; i < self->dims; ++i)
        self->shape[i] = PyLong_AsSize_t(PyList_GET_ITEM(shape, i));
    return 0;
}

static ArrayObject * Array_random(ArrayObject *self, PyObject *args) {
    PyObject *py_shape;
    if (!PyArg_ParseTuple(args, "O", &py_shape))
    { return NULL; }
    size_t dims = PyTuple_Size(py_shape);
    size_t *shape = new_size_t(dims);
    size_t size = 1;
    for (int i = 0; i < dims; ++i) {
        shape[i] = PyLong_AsSize_t(PyTuple_GET_ITEM(py_shape, i));
        size = size * shape[i];
    }
    srand(time(NULL));
    double *data = new_double(size);
    for (size_t i = 0; i < size; ++i)
        data[i] = (double)(rand()) / 10000000000;
    ArrayObject *array;
    array = PyObject_New(ArrayObject, ArrayTypePtr);
    array->data = data;
    array->size = size;
    array->dims = dims;
    array->shape = shape;
    return array;
}

static void Array_copy(ArrayObject *array1,
                       ArrayObject *array2) {
    array2->shape = new_size_t(array1->dims);
    for (size_t i = 0; i < array1->dims; ++i)
        array2->shape[i] = array1->shape[i];
    array2->data = new_double(array1->size);
    array2->size = array1->size;
    array2->dims = array1->dims;
}
static ArrayObject * Array_unary_operation(
        ArrayObject *array,
        double(*operation)(double)) {
    ArrayObject *result;
    result = PyObject_New(ArrayObject, ArrayTypePtr);
    Array_copy(array, result);
    for (size_t i = 0; i < array->size; ++i) {
        result->data[i] = operation(array->data[i]);
    }
    return result;
}
static ArrayObject * Array_binary_operation(
        ArrayObject *array1, ArrayObject *array2,
        double (*operation)(double, double)) {
    ArrayObject *result;
    result = PyObject_New(ArrayObject, ArrayTypePtr);
    if (array1->dims != 0 && array2->dims != 0) {
        Array_copy(array1, result);
        if (array1->dims == array2->dims) {
            for (size_t i = 0; i < array1->size; ++i) {
                result->data[i] = operation(array1->data[i],
                                            array2->data[i]);
            }
        } else if (array1->dims > array2->dims) {
            for (size_t i = 0; i < array1->size; i += array2->size)
                for (size_t j = 0; j < array2->size; j++)
                    result->data[i + j] = operation(array1->data[i + j],
                                                    array2->data[0 + j]);

        } else if (array1->dims < array2->dims) {
            for (size_t i = 0; i < array1->size; i++) {
                result->data[i] = array1->data[i];
                for (size_t j = 0; j < array2->size; j += array1->size) {
                    result->data[i] = operation(result->data[i + 0],
                                                array2->data[i + j]);
                }
            }
        }
    } else if (array1->dims != 0 && array2->dims == 0) {
        Array_copy(array1, result);
        for (size_t i = 0; i < array1->size; ++i) {
            result->data[i] = operation(array1->data[i],
                                        array2->data[0]);
        }
    } else if (array1->dims == 0 && array2->dims != 0) {
        Array_copy(array2, result);
        for (size_t i = 0; i < array2->size; ++i) {
            result->data[i] = operation(array1->data[0],
                                        array2->data[i]);
        }
    } else if (array1->dims == 0 && array2->dims == 0) {
        Array_copy(array2, result);
        result->data[0] = operation(array1->data[0],
                                    array2->data[0]);
    }
    return result;
}
static ArrayObject * Array_mm(
        ArrayObject *array1,
        ArrayObject *array2) {
    ArrayObject *result;
    result = PyObject_New(ArrayObject, ArrayTypePtr);
    size_t dim11 = array1->shape[0], dim12 = array1->shape[1];
    size_t dim21 = array2->shape[0], dim22 = array2->shape[1];
    result->data = new_double(dim11 * dim22);
    result->shape = new_size_t(2);
    result->size = dim11 * dim22;
    result->shape[0] = dim11;
    result->shape[1] = dim22;
    result->dims = 2;

    for (size_t i = 0; i < dim11; i++) {
        for (size_t j = 0; j < dim22; j++) {
            result->data[i * dim22 + j] = 0;
            for (size_t k = 0; k < dim21; k++) {
                result->data[i * dim22 + j] +=
                        array1->data[i * dim12 + k] *
                        array2->data[j + dim22 * k];
            }
        }
    }
    return result;
}
static ArrayObject * Array_getitem(ArrayObject *self, PyObject *args) {
    PyObject *index;
    if (!PyArg_ParseTuple(args, "O", &index))
    { return NULL; }
    size_t idx = PyLong_AsSize_t(index);
    ArrayObject *result;
    result = PyObject_New(ArrayObject, ArrayTypePtr);
    size_t size = self->size / self->shape[0];
    result->data = self->data + idx * size;
    result->shape = self->shape + 1;
    result->dims = self->dims - 1;
    result->size = size;
    return result;
}
static size_t Array_absolute_index(ArrayObject *self, size_t *relative_index) {
    size_t absolute_index = 0;
    size_t size = self->size;
    for (size_t i = 0; i < self->dims; i++) {
        size = size / self->shape[i];
        absolute_index += relative_index[i] * size;
    }
    return absolute_index;
}
static size_t* Array_relative_index(ArrayObject *self, size_t absolute_index) {
    size_t size = self->size;
    size_t* relative_index = new_size_t(self->dims);
    for (size_t i = 0; i < self->dims - 1; i++) {
        size = size / self->shape[i];
        relative_index[i] = absolute_index / size;
        absolute_index -= absolute_index / size * size;
    }
    relative_index[self->dims - 1] =
            absolute_index % self->shape[self->dims - 1];
    return relative_index;
}
static ArrayObject * Array_transpose(ArrayObject *self) {
    ArrayObject *result;
    result = PyObject_New(ArrayObject, ArrayTypePtr);
    result->data = new_double(self->size);
    result->shape = reverse(self->shape,
                            self->dims);
    result->size = self->size;
    result->dims = self->dims;
    for (size_t i = 0; i < self->size; ++i) {
        size_t *relative_index = Array_relative_index(result, i);
        relative_index = reverse(relative_index, self->dims);
        size_t absolute_index = Array_absolute_index(self, relative_index);
        result->data[i] = self->data[absolute_index];
    }
    return result;
}

static ArrayObject * Array_add(ArrayObject *self, PyObject *args) {
    PyObject *other;
    if (!PyArg_ParseTuple(args, "O", &other))
    { return NULL; }
    return Array_binary_operation(self, (ArrayObject*)other, __add__);
}
static ArrayObject * Array_sub(ArrayObject *self, PyObject *args) {
    PyObject *other;
    if (!PyArg_ParseTuple(args, "O", &other))
    { return NULL; }
    return Array_binary_operation(self, (ArrayObject*)other, __sub__);
}
static ArrayObject * Array_mul(ArrayObject *self, PyObject *args) {
    PyObject *other;
    if (!PyArg_ParseTuple(args, "O", &other))
    { return NULL; }
    return Array_binary_operation(self, (ArrayObject*)other, __mul__);
}
static ArrayObject * Array_div(ArrayObject *self, PyObject *args) {
    PyObject *other;
    if (!PyArg_ParseTuple(args, "O", &other))
    { return NULL; }
    return Array_binary_operation(self, (ArrayObject*)other, __div__);
}
static ArrayObject * Array_pow(ArrayObject *self, PyObject *args) {
    PyObject *other;
    if (!PyArg_ParseTuple(args, "O", &other))
    { return NULL; }
    return Array_binary_operation(self, (ArrayObject*)other, __pow__);
}
static ArrayObject * Array_matmul(ArrayObject *self, PyObject *args) {
    PyObject *other;
    if (!PyArg_ParseTuple(args, "O", &other))
    { return NULL; }
    return Array_mm(self, (ArrayObject *) other);
}
static ArrayObject * Array_sigmoid(ArrayObject *self) {
    return Array_unary_operation(self, sigmoid);
}
static ArrayObject * Array_tanh(ArrayObject *self) {
    return Array_unary_operation(self, tanh);
}
static ArrayObject * Array_relu(ArrayObject *self) {
    return Array_unary_operation(self, relu);
}
static ArrayObject * Array_d_sigmoid(ArrayObject *self) {
    return Array_unary_operation(self, d_sigmoid);
}
static ArrayObject * Array_d_tanh(ArrayObject *self) {
    return Array_unary_operation(self, d_tanh);
}
static ArrayObject * Array_d_relu(ArrayObject *self) {
    return Array_unary_operation(self, d_relu);
}
static PyObject * Array_sum(ArrayObject *self) {
    double result = 0;
    for (size_t i = 0; i < self->size; ++i)
        result = result + self->data[i];
    return PyFloat_FromDouble(result);
}
static PyObject * Array_mean(ArrayObject *self) {
    double result = 0;
    for (size_t i = 0; i < self->size; ++i)
        result = result + self->data[i];
    return PyFloat_FromDouble(result / self->size);
}
static PyObject * Array_max(ArrayObject *self) {
    double result = INT_MIN;
    for (size_t i = 0; i < self->size; ++i) {
        if (self->data[i] > result)
            result = self->data[i];
    }
    return PyFloat_FromDouble(result);
}
static PyObject * Array_min(ArrayObject *self) {
    double result = INT_MAX;
    for (size_t i = 0; i < self->size; ++i) {
        if (self->data[i] < result)
            result = self->data[i];
    }
    return PyFloat_FromDouble(result);
}

static PyObject * Array_get_data(ArrayObject *self) {
    PyObject *py_data = PyList_New(self->size);
    for (size_t i = 0; i < self->size; ++i) {
        PyList_SET_ITEM(py_data, i, PyFloat_FromDouble(self->data[i]));
    }
    return py_data;
}
static PyObject * Array_get_shape(ArrayObject *self) {
    PyObject *py_shape = PyList_New(self->dims);
    for (size_t i = 0; i < self->dims; ++i) {
        PyList_SET_ITEM(py_shape, i, PyFloat_FromDouble(self->shape[i]));
    }
    return py_shape;
}
static PyObject * Array_get_size(ArrayObject *self) {
    return PyLong_FromSize_t(self->size);
}
static PyObject * Array_get_dims(ArrayObject *self) {
    return PyLong_FromSize_t(self->dims);
}

static PyMemberDef Array_members[] = {
        {NULL},
};
static PyMethodDef Array_methods[] = {
        {"__add__", (PyCFunction)Array_add, METH_VARARGS, "add"},
        {"__sub__", (PyCFunction)Array_sub, METH_VARARGS, "sub"},
        {"__mul__", (PyCFunction)Array_mul, METH_VARARGS, "mul"},
        {"__truediv__", (PyCFunction)Array_div, METH_VARARGS, "div"},
        {"__pow__", (PyCFunction)Array_pow, METH_VARARGS, "pow"},
        {"__matmul__", (PyCFunction)Array_matmul, METH_VARARGS, "matmul"},
        {"sigmoid", (PyCFunction)Array_sigmoid, METH_NOARGS, "sigmoid"},
        {"tanh", (PyCFunction)Array_tanh, METH_NOARGS, "tanh"},
        {"relu", (PyCFunction)Array_relu, METH_NOARGS, "relu"},
        {"d_sigmoid", (PyCFunction)Array_d_sigmoid, METH_NOARGS, "d_sigmoid"},
        {"d_tanh", (PyCFunction)Array_d_tanh, METH_NOARGS, "d_tanh"},
        {"d_relu", (PyCFunction)Array_d_relu, METH_NOARGS, "d_relu"},
        {"transpose", (PyCFunction)Array_transpose, METH_NOARGS, "transpose"},
        {"__getitem__", (PyCFunction)Array_getitem, METH_VARARGS, "getitem"},
        {"get_data", (PyCFunction)Array_get_data, METH_NOARGS, "get_data"},
        {"get_shape", (PyCFunction)Array_get_shape, METH_NOARGS, "get_shape"},
        {"get_size", (PyCFunction)Array_get_size, METH_NOARGS, "get_size"},
        {"get_dims", (PyCFunction)Array_get_dims, METH_NOARGS, "get_dims"},
        {"random", (PyCFunction)Array_random, METH_VARARGS, "random"},
        {"mean", (PyCFunction)Array_mean, METH_NOARGS, "mean"},
        {"max", (PyCFunction)Array_max, METH_NOARGS, "max"},
        {"min", (PyCFunction)Array_min, METH_NOARGS, "min"},

        {NULL}
};

static PyTypeObject ArrayType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "cndarray.cndarray",
        .tp_doc = "cndarray objects",
        .tp_basicsize = sizeof(ArrayObject),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = Array_new,
        .tp_init = (initproc) Array_init,
        .tp_members = Array_members,
        .tp_methods = Array_methods,
};

static PyModuleDef ArrayModule = {
        PyModuleDef_HEAD_INIT,
        .m_name = "cndarray",
        .m_doc = "Example module that creates an extension type.",
        .m_size = -1
};

PyMODINIT_FUNC
PyInit_cndarray(void)
{
    PyObject *module;
    if (PyType_Ready(&ArrayType) < 0)
        return NULL;

    module = PyModule_Create(&ArrayModule);
    if (module == NULL)
        return NULL;

    Py_INCREF(&ArrayType);
    if (PyModule_AddObject(module, "cndarray", (PyObject *) &ArrayType) < 0) {
        Py_DECREF(&ArrayType);
        Py_DECREF(module);
        return NULL;
    }

    return module;
}