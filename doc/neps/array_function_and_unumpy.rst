``__array_function__`` and ``unumpy``
-------------------------------------

Strengths/advantages of ``__array_function__`` (and ``__array_ufunc__`` - we
will treat these two protocols as one in this document since they're almost
identical) are:

1. It is lightweight and has low overhead.
2. It composes well without end users having to do anything, or libraries
   having to know what array types will be present/supported.
3. It acts without any global context or registry, so debugging experience is
   nice.

Those advantages come with some tradeoffs, that are due to attaching the
dispatch logic directly to an array object.  What isn't possible with
``__array_function__``:

1. Override functions that do not take an array-like argument
2. Override anything other than functions
3. Write overrides for functions with those overrides expecting ``numpy.ndarray``
   input. Or more generally: have alternative implementations of any function
   for the same array type act as an override.
4. Write overrides outside of the package that provides the array-like object.
5. Dispatch on other (non array-like) function arguments.

In this document we will discuss each of those impossibilities, with examples
of how they are (or may become) real-world issues, and how ``unumpy`` can be
used to solve them.


Override functions that do not take an array-like argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Examples:** see NEP 30 and
`Other *_like creation functions <https://github.com/numpy/numpy/issues/14441>`__.

Also, ``numpy.random`` functions cannot be used that way.  E.g. there's a
``cupy.random.rand`` (and other functions), but those cannot be dispatched to.
Now of course for these random functions it's anyway good practice to use an
instance of a generator and use the attached methods, but (a) CuPy doesn't have
the new generators that are in NumPy 1.17.0, and (b) ``RandomState`` is also
not overridable (see next section).

**How to do it with unumpy**



Override anything other than functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Examples** include:

- classes (e.g. CuPy can't override ``np.ufunc`` with ``cupy.ufunc`` or
  ``np.random.RandomState`` with ``cupy.random.RandomState``, or ``np.r_`` with
  ``cupy.r_``)
- context managers (e.g. ``np.errstate``) TODO: more concrete need?
- constants TODO example?
- ...

**How to do it with unumpy**


Write overrides for functions with those overrides expecting `numpy.ndarray` input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Examples:**

``mkl_fft`` and ``pyfftw`` both provide alternatives for functionality in
``numpy.fft`` (as does ``scipy.fft``). While the ``numpy.fft`` functions are
decorated with ``__array_function__``, it's not possible to have those
functions dispatch to those packages.


Same for ``mkl_random``.

`opt_einsum <https://github.com/dgasmith/opt_einsum>`__ has a faster
implementation of ``np.einsum`` for ``numpy.ndarray``.

`bottleneck <https://kwgoodman.github.io/bottleneck-doc/index.html>`__ has
implementation of many of NumPy's ``nan``-functions and ``np.partition``,
``np.argpartition``.

**How to do it with unumpy**


Write overrides outside of the package that provides the array-like object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Examples:** Dask and CuPy have a namespace that is structured exactly like
NumPy's.  Other array libraries (Tensorflow, PyTorch, MXNet, XND, Xtensor,
ChainerX) may not want to do this - keeping a small core, perhaps even _only_
an n-dimensional array object, and having other packages provide functions that
are NumPy-equivalent can make a lot of sense.  ``__array_function__``
dispatching won't work for such a package structure: the
``arrayobject.__array_function__`` implementation needs to be aware of the
implementation to be able to dispatch to it.

**How to do it with unumpy**



Dispatch on other (non array-like) function arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Examples:** a package that just does (e.g.) optimized ``float32`` and
``float64`` implementations of an algorithm would want dispatch to it to
only happen for those dtypes, and stay with the NumPy implementation for other
dtypes. Bottleneck is such an example.  For other dtypes it now does::

    def nanargmin(a, axis=None):
        "Slow nanargmin function used for unaccelerated dtypes."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanargmin(a, axis=axis)

There are quite a few of these functions in Bottleneck; they take more code to
implement, and the roundtrip function calls add unnecessary overhead.

Note that dispatching on dtype is not unique to Bottleneck; in PyTorch for
example dispatching on dtype to specialized kernels is very common.

**How to do it with unumpy**


What happens if ``__array_function__`` and ``unumpy`` are both active
---------------------------------------------------------------------

*Note: both being active occurs if a user does ``import unumpy as np`` and uses
functions from that namespace, or if we would add ``unumpy`` to NumPy and make
it default. For the discussion in this section those two situations are the
same.*

By default, ``unumpy`` will only set NumPy as the global backend. So then a
call like ``np.mean(x)`` will go through ``unumpy`` first, dispatch to its
``numpy_backend``, and then ``__array_function__`` kicks in and dispatch based
on ``x.__array_function__`` will happen.

In case a user has explicitly chosen a backend *other* than ``numpy_backend``,
then ``unumpy`` will dispatch to that and the ``__array_function__`` dispatch
won't happen.

Recommended usage for array library authors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Library authors should write code that makes use of ``__array_function__`` if
possible, and ``unumpy`` if needed. Example::

    TODO  # only sensible to use unumpy with the proposed ``determine_backend`` I think


Recommended usage for library authors that support multiple array libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Recommended usage for end users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO


Random notes
------------

Principle: adding a new function to NumPy should only be done if that function
makes sense for NumPy itself.  Adding new functions purely to work around the
limitations of ``__array_function__`` is not desirable.

