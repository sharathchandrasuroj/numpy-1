===============================================================
NEP XXXX — Using SIMD optimization instructions for performance
===============================================================

:Author: Sayed Adel, Matti Picus, Ralf Gommers
:Status: Draft
:Type: Standards
:Created: 2019-11-25
:Resolution: none


Abstract
--------

While compilers are getting better at using hardware-specific routines to
optimize code, they sometimes do not produce optimal results. Also, we would
like to be able to copy binary C-extension modules from one machine to another
without recompiling.

We have a mechanism in the ufunc machinery to choose optimal loops. This NEP
proposes a mechanism to build on that for many more features and architectures.
The steps would be:

- Establish a baseline of CPU features for minimal support
- Write explicit code to take advantage of well-defined, architecture-agnostic,
  universal intrisics which capture features available across architectures.
- Capture those universal intrisics in a set of C macros that at compile time
  would build code paths for each feature from the baseline up to the maximum
  set of features available on that architecture.
- At runtime, discover which CPU features are available, and choose from among
  the possible code paths accordingly.


Motivation and Scope
--------------------

Traditionally NumPy has counted on the compilers to generate optimal code.
Recently there were discussions around whether this is `good enough`_ or
if hand-written code is needed. Some architecture-specific code was added to
NumPy for `fast routines`_ on x86 in ufuncs, using the loop-resolution routines
to choose the correct loop for the architecture. However the code is not
generic and does not generalize to other architectures.

It would be nice if the universal intrinsics would be available to other
libraries like SciPy or Astropy that also build ufuncs, but that is not an
explicit goal of the first implementation of this NEP.


Usage and Impact
----------------

*This section describes how users of NumPy will use features described in this
NEP. It should be comprised mainly of code examples that wouldn't be possible
without acceptance and implementation of this NEP, as well as the impact the
proposed changes would have on the ecosystem. This section should be written
from the perspective of the users of NumPy, and the benefits it will provide
them; and as such, it should include implementation details only if
necessary to explain the functionality.*



Binary releases - wheels on PyPI and conda packages
```````````````````````````````````````````````````

- What optimizations will be enabled, and which are used at runtime?
- Is there a change in size of binaries?


Source builds
`````````````

- Setting the baseline and set of runtime-dispatchable ISA extensions
- Behavior when compiler or hardware doesn't support a requested ISA extension


How to run benchmarks to assess performance benefits
````````````````````````````````````````````````````

Adding more code which use intrinsics will make the code harder to maintain.
Therefore, such code should only be added if it yields a significant
performance benefit. Assessing this performance benefit can be nontrivial.
To aid with this, the implementation for this NEP will add a way to select
which instruction sets can be used at *runtime* via an environment variable
(name TBD).


Diagnostics
```````````
- How does a user know what optimizations are enabled for the installed NumPy?
  Add to `show_config()` output? What about actual runtime capability on the
  installed system?


Workflow for adding a new CPU architecture-specific optimization
````````````````````````````````````````````````````````````````

NumPy will always have a baseline C implementation for any code that may be
a candidate for SIMD vectorization.  Now if a contributor wants to add SIMD
support for some architecture (typically the one of most interest to them),
this is the proposed workflow:

TODO (see https://github.com/numpy/numpy/pull/13516#issuecomment-558859638,
needs to be worked out more)



Backward compatibility
----------------------

There should be no impact on backwards compatibility.


Detailed description
--------------------

*This section should provide a detailed description of the proposed change.
It should include examples of how the new functionality would be used,
intended use-cases and pseudo-code illustrating its use.*

TODO: status today - what instructions are used at build time (SSE2/3 and
AVX/AVX512 for XXX functionality) and at runtime (some but less, see
``loops.c.src``)


Related Work
------------

*This section should list relevant and/or similar technologies, possibly in other
libraries. It does not need to be comprehensive, just list the major examples of
prior and relevant art.*

- OpenCV
- PIXMAX
- Eigen
- xsimd_


Implementation
--------------

*This section lists the major steps required to implement the NEP.  Where
possible, it should be noted where one step is dependent on another, and which
steps may be optionally omitted.  Where it makes sense, each step should
include a link to related pull requests as the implementation progresses.

Any pull requests or development branches containing work on this NEP should
be linked to from here.  (A NEP does not need to be implemented in a single
pull request if it makes sense to implement it in discrete phases).*

Current PRs:

- `gh-13421 improve runtime detection of CPU features <https://github.com/numpy/numpy/pull/13421>`_
- `gh-13516: enable multi-platform SIMD compiler optimizations <https://github.com/numpy/numpy/pull/13516>`_

**Let's leave description of this out for now. Only do that once the questions
in the sections above are answered.**


Alternatives
------------

*If there were any alternative solutions to solving the same problem, they should
be discussed here, along with a justification for the chosen approach.*

A proposed alternative in gh-13516_ is a per CPU architecture implementation of
SIMD code (e.g., have `loops.avx512.c.src`, `loops.avx2.c.src`, `loops.sse.c.src`,
`loops.vsx.c.src`, `loops.neon.c.src`, etc.). This is more similar to what
OpenCV and PIXMAX do. There's a lot of duplication here though, it is likely
much harder to maintain.


Discussion
----------

*This section may just be a bullet list including links to any discussions
regarding the NEP:

- This includes links to mailing list threads or relevant GitHub issues.*



References and Footnotes
------------------------

.. _`good enough`: https://github.com/numpy/numpy/pull/11113
.. _`fast routines`: https://github.com/numpy/numpy/pulls?q=is%3Apr+avx512+is%3Aclosed

.. [1] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. _`xsimd`: https://xsimd.readthedocs.io/en/latest/


Copyright
---------

This document has been placed in the public domain. [1]_
